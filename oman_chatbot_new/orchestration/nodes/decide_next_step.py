# decide_next_step.py
class DecideNextStep:
    def __init__(self, threshold=3):
        self.threshold = threshold

    def run(self, state: dict) -> dict:
        relevant_count = state["keys"].get("relevant_count", 0)
        did_transform = state["keys"].get("did_transform", False)

        # Decide a route
        if relevant_count < self.threshold:
            route = "transform_query" if not did_transform else "web_search"
        else:
            route = "generate"

        # Store the route in the state
        state["keys"]["_next_route"] = route
        return state
