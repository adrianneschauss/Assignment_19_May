import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

#you need to keep the motor spinning with a ratio of voltage to frequency that is constant, so if you slow down the motor the voltage needs to proportionaly drop
#if you keep the motor voltage too high relative to the frequency, you boost the magnetic field because more time for the coils togenerate magnetic field,
# then excessive current is drawn and the motor burns out after overheating
#if you increase the frequency but not the torque then the magnetic flux is too small for the motor to hold the load and the required torque will not be high eniugh
#if you suddenly increase the load, then the current will icnrease to try and counter act the increase in the voltage difference, i.e you get negative voltage added to your original voltage
#more increase in voltage for the same amount of load means that less current needs to flow which is more efficient
@dataclass
class VFD:
    """
    Simple educational model of a variateur / VFD.

    It receives a 0-10 V command and converts it into:
    - target frequency
    - ramped output frequency
    - output voltage using V/f control
    """

    f_nominal: float = 50.0          # Hz
    v_nominal: float = 400.0        # V line-line RMS
    f_max: float = 50.0             # Hz
    ramp_rate: float = 20.0         # Hz per second
    current_frequency: float = 0.0  # Hz

    def command_voltage_to_frequency(self, command_voltage):
        """
        Maps a 0-10 V analog command to 0-f_max Hz.

        Example:
        0 V  -> 0 Hz
        5 V  -> 25 Hz
        10 V -> 50 Hz
        """

        command_voltage = np.clip(command_voltage, 0.0, 10.0)
        return (command_voltage / 10.0) * self.f_max

    def ramp_frequency(self, target_frequency, dt):
        """
        The drive does not jump instantly from one frequency to another.
        It ramps smoothly to protect the motor and mechanics.
        """

        max_change = self.ramp_rate * dt
        error = target_frequency - self.current_frequency

        if error > max_change:
            self.current_frequency += max_change
        elif error < -max_change:
            self.current_frequency -= max_change
        else:
            self.current_frequency = target_frequency

        return self.current_frequency

    def voltage_from_frequency(self, frequency):
        """
        V/f control.

        Below nominal frequency:
            voltage is proportional to frequency

        At 50 Hz:
            voltage reaches nominal voltage

        Example:
            50 Hz -> 400 V
            25 Hz -> 200 V
            10 Hz -> 80 V
        """

        if frequency <= 0:
            return 0.0

        voltage = self.v_nominal * (frequency / self.f_nominal)

        return min(voltage, self.v_nominal)

    def update(self, command_voltage, dt):
        """
        Full VFD update:
        1. Read command voltage
        2. Convert it to target frequency
        3. Ramp toward target frequency
        4. Compute output voltage
        """

        target_frequency = self.command_voltage_to_frequency(command_voltage)
        output_frequency = self.ramp_frequency(target_frequency, dt)
        output_voltage = self.voltage_from_frequency(output_frequency)

        return output_frequency, output_voltage


@dataclass
class ThreePhaseInductionMotor:
    """
    Very simplified induction motor model.

    This is NOT a full electromagnetic model.
    It just shows the main idea:

        frequency -> rotating magnetic field speed -> motor shaft speed

    The motor speed follows the synchronous speed with some slip and delay.
    """

    poles: int = 4
    slip: float = 0.03
    mechanical_time_constant: float = 0.6  # seconds
    speed_rpm: float = 0.0

    def synchronous_speed_rpm(self, frequency):
        """
        Synchronous speed formula:

            n_sync = 120 * f / poles

        For a 4-pole motor:
            50 Hz -> 1500 rpm theoretical
            real speed maybe ~1450 rpm because of slip
        """

        return 120.0 * frequency / self.poles

    def update(self, frequency, dt):
        """
        Motor speed follows the magnetic field speed, but not instantly.

        The target mechanical speed is slightly lower than synchronous speed
        because of slip.
        """

        n_sync = self.synchronous_speed_rpm(frequency)
        target_speed = n_sync * (1.0 - self.slip)

        # First-order mechanical response
        self.speed_rpm += (target_speed - self.speed_rpm) * dt / self.mechanical_time_constant

        return self.speed_rpm, n_sync


def three_phase_sine_wave(frequency, voltage_ll_rms, duration=0.08, dt=1e-5):
    """
    Generates ideal three-phase sine wave voltages.

    This is the clean fundamental waveform the motor approximately sees.
    The real VFD output is PWM, but the motor windings filter it.
    """

    t = np.arange(0, duration, dt)

    if frequency <= 0 or voltage_ll_rms <= 0:
        return t, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)

    # Convert line-line RMS voltage to phase peak voltage
    # V_phase_rms = V_line_line_rms / sqrt(3)
    # V_phase_peak = sqrt(2) * V_phase_rms
    v_phase_peak = np.sqrt(2) * voltage_ll_rms / np.sqrt(3)

    va = v_phase_peak * np.sin(2 * np.pi * frequency * t)
    vb = v_phase_peak * np.sin(2 * np.pi * frequency * t - 2 * np.pi / 3)
    vc = v_phase_peak * np.sin(2 * np.pi * frequency * t + 2 * np.pi / 3)

    return t, va, vb, vc


def conceptual_pwm(frequency, voltage_ratio, duration=0.04, dt=2e-6, carrier_frequency=3000):
    """
    Conceptual PWM generation.

    A real VFD uses fast power switches.
    It compares three sine references with a triangular carrier.

    This function shows the principle only.
    """

    t = np.arange(0, duration, dt)

    if frequency <= 0:
        return t, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)

    # Modulation index between 0 and 1
    m = np.clip(voltage_ratio, 0.0, 1.0)

    ref_a = m * np.sin(2 * np.pi * frequency * t)
    ref_b = m * np.sin(2 * np.pi * frequency * t - 2 * np.pi / 3)
    ref_c = m * np.sin(2 * np.pi * frequency * t + 2 * np.pi / 3)

    # Triangle carrier between -1 and +1
    carrier = 2 * np.abs(2 * ((t * carrier_frequency) % 1) - 1) - 1

    pwm_a = ref_a > carrier
    pwm_b = ref_b > carrier
    pwm_c = ref_c > carrier

    return t, pwm_a.astype(float), pwm_b.astype(float), pwm_c.astype(float)


# ------------------------------------------------------------
# Main simulation
# ------------------------------------------------------------

dt = 0.001
simulation_time = 8.0
time = np.arange(0, simulation_time, dt)

vfd = VFD(
    f_nominal=50.0,
    v_nominal=400.0,
    f_max=50.0,
    ramp_rate=15.0
)

motor = ThreePhaseInductionMotor(
    poles=4,
    slip=0.03,
    mechanical_time_constant=0.8
)

command_voltage_log = []
frequency_log = []
voltage_log = []
motor_speed_log = []
sync_speed_log = []

for t in time:
    # Example command profile from PLC/HMI/potentiometer
    if t < 1.0:
        command_voltage = 0.0      # stopped
    elif t < 3.5:
        command_voltage = 6.0      # 60% speed command
    elif t < 5.5:
        command_voltage = 10.0     # full speed command
    else:
        command_voltage = 3.0      # slow down to 30%

    frequency, voltage = vfd.update(command_voltage, dt)
    motor_speed, sync_speed = motor.update(frequency, dt)

    command_voltage_log.append(command_voltage)
    frequency_log.append(frequency)
    voltage_log.append(voltage)
    motor_speed_log.append(motor_speed)
    sync_speed_log.append(sync_speed)


# ------------------------------------------------------------
# Plot drive and motor behavior
# ------------------------------------------------------------

plt.figure(figsize=(12, 9))

plt.subplot(4, 1, 1)
plt.plot(time, command_voltage_log)
plt.ylabel("Command [V]")
plt.title("PLC command to variateur")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(time, frequency_log)
plt.ylabel("Frequency [Hz]")
plt.title("VFD output frequency after ramp")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(time, voltage_log)
plt.ylabel("Voltage [V]")
plt.title("V/f law: output voltage follows frequency")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(time, sync_speed_log, label="Synchronous speed")
plt.plot(time, motor_speed_log, label="Motor shaft speed")
plt.ylabel("Speed [rpm]")
plt.xlabel("Time [s]")
plt.title("Motor speed response")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Generate three-phase waveform at final operating point
# ------------------------------------------------------------

final_frequency = frequency_log[-1]
final_voltage = voltage_log[-1]

t_wave, va, vb, vc = three_phase_sine_wave(
    frequency=final_frequency,
    voltage_ll_rms=final_voltage,
    duration=0.12,
    dt=1e-5
)

plt.figure(figsize=(12, 5))
plt.plot(t_wave, va, label="Phase A")
plt.plot(t_wave, vb, label="Phase B")
plt.plot(t_wave, vc, label="Phase C")
plt.title(f"Ideal three-phase output at {final_frequency:.1f} Hz")
plt.xlabel("Time [s]")
plt.ylabel("Phase voltage [V]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Show conceptual PWM switching
# ------------------------------------------------------------

voltage_ratio = final_voltage / vfd.v_nominal

t_pwm, pwm_a, pwm_b, pwm_c = conceptual_pwm(
    frequency=final_frequency,
    voltage_ratio=voltage_ratio,
    duration=0.04,
    dt=2e-6,
    carrier_frequency=3000
)

plt.figure(figsize=(12, 5))
plt.plot(t_pwm, pwm_a + 2, label="PWM A")
plt.plot(t_pwm, pwm_b + 1, label="PWM B")
plt.plot(t_pwm, pwm_c, label="PWM C")
plt.title("Conceptual PWM switching inside the variateur")
plt.xlabel("Time [s]")
plt.ylabel("Switching state")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()