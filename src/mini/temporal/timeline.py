from dataclasses import dataclass

from mini.temporal.dopesheet import Dopesheet, Step
from mini.temporal.transitions import SmoothProp


@dataclass
class State:
    step: int
    phase: str
    actions: list[str]
    props: dict[str, float]
    is_phase_start: bool
    is_phase_end: bool


class Timeline:
    """
    Evolves property values over time.

    Whereas the Dopesheet defines the properties and their keyframes,
    the Timeline is responsible for interpolating between those keyframes
    and updating the properties at each step.
    """

    props: dict[str, SmoothProp]
    _step: int
    _max_steps: int

    def __init__(self, dopesheet: Dopesheet):
        """Initialize the timeline."""
        self.dopesheet = dopesheet
        self._max_steps = len(self.dopesheet)

        # Get the initial values for each property from the dopesheet
        initial_values = dopesheet.get_initial_values()

        # Create SmoothProp instances with appropriate initial values
        self.props = {}
        for prop in dopesheet.props:
            # Use the initial value if available, otherwise default to 0.0
            initial_value = initial_values.get(prop, 0.0)
            self.props[prop] = SmoothProp(value=initial_value)

        self._step = 0
        # Set things in motion
        self._process_keyframes()

    def __len__(self) -> int:
        return self._max_steps

    def _process_keyframes(self) -> None:
        """Process keyframes at the current step."""
        current_step: Step = self.dopesheet[self._step]
        for key in current_step.keyed_props:
            # Update the property with the new target value and duration. These may be None if there are no more keyframes, but that's fine because SmoothProp will interpret that as "no change".
            self.props[key.prop].set(value=key.next_value, duration=key.duration)

    def step(self) -> State:
        """Advance the timeline by one step."""
        if self._step >= self._max_steps:
            raise IndexError('Timeline has reached the end.')

        self._step += 1
        for prop in self.props.values():
            prop.step(1.0)
        self._process_keyframes()
        return self.state

    @property
    def state(self) -> State:
        """Get the current state of the timeline."""
        static_info = self.dopesheet[self._step]
        props = {prop: self.props[prop].value for prop in self.props}
        return State(
            step=self._step,
            phase=static_info.phase,
            actions=static_info.actions,
            props=props,
            is_phase_start=static_info.is_phase_start,
            is_phase_end=static_info.is_phase_end,
        )
