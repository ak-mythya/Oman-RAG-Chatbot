class DecideNextStep:
    def __init__(self, threshold=3):
        self.threshold = threshold

    def run(self, state: dict) -> dict:
        sub_query_mapping = state["keys"].get("sub_query_mapping", {})
        classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])

        # Initialize list for routes per sub-query
        sub_query_routes = []

        for sq_data in classified_sub_queries:
            classification = sq_data.get("classification", "out-of-scope")
            did_transform = sq_data.get("needs_transformation", False)  # Use transformation flag per sub-query
            relevant_count = sq_data.get("relevant_count", 0)  # Relevant count could be per sub-query

            if classification == "in-scope":
                if relevant_count < self.threshold:
                    route = "transform_query" if not did_transform else "web_search"
                else:
                    route = "generate"
            else:
                # If out-of-scope or general, no retrieval or transformation needed
                route = "generate"  # May require adjusting if the out-of-scope case has different needs

            sub_query_routes.append(route)

        # Store the routes in the state (can decide how to handle multiple sub-query routes)
        state["keys"]["_next_route"] = sub_query_routes
        return state
