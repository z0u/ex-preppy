class Timeline:
    def step(self, n=1):
        """Advance the timeline by n steps."""

    def get_props(self) -> dict[str, float]:
        """Get the property values at the current time."""

    def get_actions(self) -> list[str]:
        """Get the actions to be performed at the current time."""

    def add_action(self, action: str, t: int):
        """Add an action to the timeline."""

    def add_prop(self, name: str):
        pass
