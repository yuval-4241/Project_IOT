from IOT_EX import *
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


#Runs 30 simulation runs for each algorithm and parameter p, then calculates the average costs over time.
def average_costs_over_runs_shared_problems(
    ps=[0.2, 0.7, 1.0],
    k=0.25,
    problem_type='general',
    max_iterations=50,
    num_runs=10,
    algorithms = ['DSA', 'MGM', 'MGM2']
):

    algorithms_with_p = {'DSA'}
    results = {
        alg: {p: [] for p in (ps if alg in algorithms_with_p else [None])}
        for alg in algorithms
    }

    for run in range(num_runs):
        creator = CreateEnvironment(num_agents=30,problem_type=problem_type, seed=run)
        agents = creator.create_agents()
        creator.connect_agents(k=k, seed=run)

        for alg in algorithms:
            ps_to_iterate = ps if alg in algorithms_with_p else [None]

            for p in ps_to_iterate:
                copied_agents = deepcopy_agents(list(agents.values()))
                env = DCOPEnvironment(copied_agents)
                sim = Simulator(env)

                sim.run(
                    p=p if p is not None else 1.0,
                    algorithm=alg,
                    max_iterations=max_iterations
                )

                results[alg][p].append(sim.costs_over_time)




    # return the avf from the array
    return {
        alg: {
            p: np.mean(np.array(results[alg][p]), axis=0)
            for p in results[alg]
        }
        for alg in algorithms
    }


#Makes a full copy of all agents so each simulation run uses fresh, independent agents with no shared data.
def deepcopy_agents(agents_list):
    from copy import deepcopy
    id_to_agent = {}

    for agent in agents_list:
        new_agent = Agent(
            agent_id=agent.agent_id,
            domain=deepcopy(agent.domain),
            problem_type=deepcopy(agent.problem_type),
            environment=None  # נשים later
        )
        new_agent.value = agent.value
        new_agent.cost_tables = deepcopy(agent.cost_tables)
        new_agent.neighbors = list(agent.neighbors)  # אפשר לשים כאן כבר
        id_to_agent[new_agent.agent_id] = new_agent

    agents_dict = id_to_agent
    env = DCOPEnvironment(agents_dict)

    for agent in agents_dict.values():
        agent.environment = env

    return agents_dict





def plot_algorithms_for_k_fixed_problems(k, problem_type='general', save_as=None):
    ps = [0.2, 0.7, 1.0]
    algorithms = ['DSA', 'MGM', 'MGM2']  # הוספנו את MGM2
    max_iterations = 50

    results = average_costs_over_runs_shared_problems(
        ps=ps,
        k=k,
        problem_type=problem_type,
        max_iterations=max_iterations,
        algorithms=algorithms
    )

    plt.figure(figsize=(12, 6))

    for alg in algorithms:
        if alg == 'DSA':
            for p in ps:
                label = f"{alg} (p={p})"
                plt.plot(range(max_iterations + 1), results[alg][p], label=label)
        else:
            # עבור MGM ו־MGM2 – ריצה אחת בלבד (עם p=None)
            label = alg
            plt.plot(range(max_iterations + 1), results[alg][None], label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Average Global Cost")
    plt.title(f"Comparison of Algorithms on k={k}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as)
    else:
        plt.show()




def run_all_dcop_algorithms():#Runs the simulations and shows graphs for different problem settings and algorithms.

    plot_algorithms_for_k_fixed_problems(k=0.25, problem_type='general')
    plot_algorithms_for_k_fixed_problems(k=0.75, problem_type='general')
    plot_algorithms_for_k_fixed_problems(k=0.1, problem_type='coloring')


run_all_dcop_algorithms()