# Inspired by: https://developers.google.com/optimization/pack/bin_packing#example
from ortools.linear_solver import pywraplp


class AspectFieldPacker:
    """
    Efficiently packs aspect-based filters into eBay's maximum query set size
    of 10000 results.

    Used for the full/non-incremental queries described in the paper.
    """

    def __init__(self, max_bin_size=10000):
        self.solver = pywraplp.Solver.CreateSolver("SCIP")
        self.max_bin_size = max_bin_size

    def pack(self, aspect_distribution: dict):
        """
        Given an aspectDistribution from eBay's API, pack the values of the
        distribution into bins up to size 10000 in the minimum number of bins.

        Usage is to query all available data with the minimum API calls.

        :param aspect_distribution: dict following API format {
            localizedAspectName: str,
            aspectValueDistributions: [
                {
                    localizedAspectValue: str,
                    refinementHref: str,
                    matchCount: int
                }
            ]
        }

        Returns a list of dictionaries, each representing a bin with the following structure:
        {
            "aspects": [int],
            "total_weight": int,
            "aspectNames": [str],
            "weights": [int]
        }
        """
        # Format the data
        data = {
            "counts": [],
            "aspects": [],
            "aspectNames": [],
            "bins": [],
            "bin_capacity": self.max_bin_size,
        }
        for i, aspect in enumerate(aspect_distribution["aspectValueDistributions"]):
            data["counts"].append(aspect["matchCount"])
            data["aspectNames"].append(aspect["localizedAspectValue"])
            data["aspects"].append(i)
        data["bins"] = list(range(len(data["counts"])))
        # Create the variables
        x = {}
        for i in data["aspects"]:
            for j in data["bins"]:
                x[(i, j)] = self.solver.IntVar(0, 1, f"x_{i}_{j}")
        y = {}
        for j in data["bins"]:
            y[j] = self.solver.IntVar(0, 1, f"y[{j}]")
        # Create the constraints
        for i in data["aspects"]:  # One item per bin
            self.solver.Add(sum(x[i, j] for j in data["bins"]) == 1)
        for j in data["bins"]:  # Bins can't have more than their capacity
            self.solver.Add(
                sum(x[(i, j)] * data["counts"][i] for i in data["aspects"])
                <= y[j] * data["bin_capacity"]
            )
        # Create the objective
        self.solver.Minimize(self.solver.Sum([y[j] for j in data["bins"]]))
        # Solve
        status = self.solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            results = []
            for j in data["bins"]:
                if y[j].solution_value() == 1:
                    bin_aspects = []
                    bin_weight = 0
                    for i in data["aspects"]:
                        if x[(i, j)].solution_value() > 0:
                            bin_aspects.append(data["aspects"][i])
                            bin_weight += data["counts"][i]
                    if bin_aspects:
                        results.append(
                            {
                                "aspects": bin_aspects,
                                "total_weight": bin_weight,
                                "aspectNames": [
                                    data["aspectNames"][i] for i in bin_aspects
                                ],
                                "weights": [data["counts"][i] for i in bin_aspects],
                            }
                        )
            return results
        else:
            raise Exception("Failed to solve the packing problem.")
