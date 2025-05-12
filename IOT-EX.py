import random
import networkx as nx
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, agent_id, domain, problem_type='general'):
        self.agent_id = agent_id
        self.domain = domain[:]
        self.problem_type = problem_type
        self.value = random.choice(self.domain)
        self.neighbors = []  # רק רשימת IDs של שכנים
        self.cost_tables = {}  # עלויות מול שכנים
        self.inbox = []  # תיבת דואר - הודעות נכנסות באיטרציה הנוכחית

        # למימוש MGM-2
        self.is_proposer = False
        self.partner_id = None
        self.proposed_assignment = None #the best value for each
        self.shared_lr = None
        self.partner_approved = False
        self.received_proposals = []
        self.lr_from_neighbors = {}
        self.outbox = []  # תיבת דואר יוצאת – הודעות שישלח בפאזות

    def receive_messages(self, messages):
        self.inbox = messages  # כל ההודעות שקיבלתי באיטרציה

    def compute_cost(self, my_value, received_messages):
        """חישוב עלות נוכחית או עבור ערך אלטרנטיבי."""
        total_cost = 0
        for msg in received_messages:
            if 'value' not in msg:
                continue  # דלג על הודעות שלא מכילות ערך

            neighbor_id = msg['sender']
            neighbor_value = msg['value']
            cost_table = self.cost_tables.get(neighbor_id)
            if cost_table is not None:
                cost = cost_table.get((my_value, neighbor_value), 0)
                total_cost += cost
            else:
                # אם אין טבלה בכלל לשכן – מתעלמים
                continue
        return total_cost

    def run_mgm_phase1(self):
        # סינון הודעות עם value בלבד
        value_messages = [msg for msg in self.inbox if 'value' in msg]

        # חישוב עלות נוכחית
        current_cost = self.compute_cost(self.value, value_messages)

        # חיפוש ערך משופר
        best_value = self.value
        best_cost = current_cost

        for value in self.domain:
            if value == self.value:
                continue
            cost = self.compute_cost(value, value_messages)
            if cost < best_cost:
                best_value = value
                best_cost = cost

        gain = current_cost - best_cost

        self.last_decision = {
            'agent_id': self.agent_id,
            'gain': gain,
            'new_value': best_value,
            'current_value': self.value
        }

        gain_message = {
            'sender': self.agent_id,
            'gain': gain
        }

        return gain_message

    def run_mgm_phase2(self, neighbor_gains):
        my_gain = self.last_decision['gain']
        my_id = self.agent_id

        i_am_max = True
        for msg in neighbor_gains:
            if msg['gain'] > my_gain or (msg['gain'] == my_gain and msg['sender'] < my_id):
                i_am_max = False
                break

        changed = False
        if i_am_max and my_gain > 0 and self.last_decision['new_value'] != self.value:
            self.value = self.last_decision['new_value']
            changed = True

        return changed

    def run_dsa(self, p):
        outgoing_messages = []

        # חישוב העלות הנוכחית
        current_cost = self.compute_cost(self.value, self.inbox)
        best_value = self.value
        best_cost = current_cost

        # חיפוש ערך שיביא לעלות טובה יותר
        for value in self.domain:
            if value == self.value:
                continue
            cost = self.compute_cost(value, self.inbox)


            if cost < best_cost:
                best_value = value
                best_cost = cost

        # החלטה האם לשנות ערך לפי הסתברות p
        changed = False
        if best_value != self.value and random.random() < p:
            self.value = best_value
            changed = True

        # שליחת הודעה לשכנים עם הערך החדש/נוכחי
        for neighbor_id in self.neighbors:
            outgoing_messages.append({
                'sender': self.agent_id,
                'receiver': neighbor_id,
                'value': self.value
            })

        return outgoing_messages, changed

    def compute_best_joint_assignment(
            self_value, self_domain, self_neighbors, self_cost_tables,
            proposer_value, proposer_domain, proposer_neighbors, proposer_cost_tables,
            neighbor_values
    ):
        """
        מחשבת את ההשמה המשותפת הטובה ביותר (self × proposer)
        עבור כל קומבינציה אפשרית של ערכי self ו־proposer,
        תוך שימוש בערכים הנוכחיים של השכנים.

        מחזירה את ההשמה שגורמת ל-LR המשותף הגבוה ביותר.
        """

        def compute_cost_MGM2(val, neighbors, tables):
            total = 0
            for neighbor_id, neighbor_val in neighbors.items():
                cost_table = tables.get(neighbor_id)
                if cost_table:
                    total += cost_table.get((val, neighbor_val), 0)
            return total

        def compute_total_cost(val_self, val_proposer):
            cost_self = compute_cost_MGM2(val_self, self_neighbor_values, self_cost_tables)
            cost_between = self_cost_tables.get('proposer', {}).get((val_self, val_proposer), 0)
            cost_self -= cost_between  # נטרול הקשת מהחישוב של self

            cost_proposer = compute_cost_MGM2(val_proposer, proposer_neighbor_values, proposer_cost_tables)
            cost_proposer -= cost_between

            return cost_self + cost_proposer + cost_between

        # פירוק שכנים למילונים רלוונטיים
        self_neighbor_values = {nid: neighbor_values[nid] for nid in self_neighbors if nid in neighbor_values}
        proposer_neighbor_values = {nid: neighbor_values[nid] for nid in proposer_neighbors if nid in neighbor_values}

        # עלות נוכחית להשמה הקיימת
        current_cost = compute_total_cost(self_value, proposer_value)

        # חיפוש קומבינציה אופטימלית
        best_total_cost = current_cost
        best_self_value = self_value
        best_proposer_value = proposer_value

        for val_self in self_domain:
            for val_proposer in proposer_domain:
                cost = compute_total_cost(val_self, val_proposer)
                print(f"cost is = {cost}")
                if cost < best_total_cost:
                    best_total_cost = cost
                    best_self_value = val_self
                    best_proposer_value = val_proposer

        joint_lr = current_cost - best_total_cost

        print(f"[JOINT ASSIGNMENT] current = {current_cost}, best_cost = {best_total_cost}, joint_lr = {joint_lr:.2f}")

        return {
            'joint_lr': joint_lr,
            'best_self_value': best_self_value,
            'best_proposer_value': best_proposer_value
        }

    def phase1_propose(self):
        PROPOSER_PROBABILITY = 0.5
        self.is_proposer = random.random() < PROPOSER_PROBABILITY
        if self.is_proposer and self.neighbors:
            self.partner_id = random.choice(self.neighbors)
            message = {
                'type': 'proposal',
                'value': self.value,
                'domain': self.domain,
                'neighbors': self.neighbors,
                'cost_tables': {
                    neighbor: self.cost_tables.get(neighbor, {}) for neighbor in self.neighbors
                },
                'sender': self.agent_id,
                'to': self.partner_id
            }
            self.outbox.append(message)


    def phase2_respond(self):
        self.partner_approved = False
        proposals = [msg for msg in self.inbox if msg['type'] == 'proposal']
        if not proposals:
            return
        selected = random.choice(proposals)  # שלב 1
        proposer_id = selected['sender']
        neighbor_values = {
            msg['sender']: msg['value']
            for msg in self.inbox if msg.get('type') == 'value_broadcast'
        }

        # שלב 2–3: חישוב LR והשמה אופטימלית
        result = self.compute_best_joint_assignment(

            self_domain=self.domain,
            self_neighbors=self.neighbors,
            self_cost_tables=self.cost_tables,
            proposer_value=selected['value'],
            proposer_domain=selected['domain'],
            proposer_neighbors=selected['neighbors'],
            proposer_cost_tables=selected['cost_tables'],
            neighbor_values=neighbor_values  # 🔁 זה היה חסר!
        )
        # שלב 4
        self.partner_id = proposer_id
        self.proposed_assignment = result['best_self_value']
        self.shared_lr = result['joint_lr']
        self.partner_approved = True

        # שלב 5: שליחת תגובה
        self.outbox.append({
            'type': 'proposal_response',
            'sender': self.agent_id,
            'to': proposer_id,
            'joint_lr': result['joint_lr'],
            'best_self_assignment': result['best_self_value'],
            'best_partner_assignment': result['best_proposer_value']
        })

    def phase3_send_lr(self):
        # איפוס ברירת מחדל
        self.partner_approved = False
        self.shared_lr = None
        self.proposed_assignment = None

        # ניסיון לקבל תגובת שותף להצעה
        responses = [msg for msg in self.inbox if msg['type'] == 'proposal_response']
        if responses:
            response = responses[0]
            self.proposed_assignment = response['best_partner_assignment']
            partner_assignment = response['best_self_assignment']
            self.shared_lr = response['joint_lr']
            self.partner_approved = True
            lr = self.shared_lr
        else:
            # אין שותף – נשתמש ב־compute_local_lr שמחשב גם את ההשמה
            lr = self.compute_local_lr()
            if hasattr(self, 'last_decision'):
                self.proposed_assignment = self.last_decision.get('new_value', self.value)

        # שליחת LR (מתבצעת תמיד)


        for neighbor in self.neighbors:
            self.outbox.append({
                'type': 'lr_notification',
                'lr': lr,
                'sender': self.agent_id,
                'to': neighbor
            })

        return self.partner_approved  # מחזיר מידע האם הסוכן בזוג

    def phase4_check_if_best(self):
        # שלב 1: קבלת ה־LR מכל השכנים
        lrs = [msg for msg in self.inbox if msg['type'] == 'lr_notification']
        self.lr_from_neighbors = {msg['sender']: msg['lr'] for msg in lrs}

        # שלב 2: חישוב ה־LR שלי – זוגי אם אני בזוג, אחרת לוקאלי
        my_lr = self.shared_lr if self.partner_approved else self.compute_local_lr()

        # שלב 3: קביעה האם אני הכי טוב בשכונה
        max_neighbor_lr = max(self.lr_from_neighbors.values(), default=float('-inf'))
        if my_lr > 0:
            # שובר שוויון לפי agent_id אם יש כמה עם אותו LR
            top_agents = [agent for agent, lr in self.lr_from_neighbors.items() if lr == max_neighbor_lr]
            self.best_in_group = (
                    my_lr > max_neighbor_lr or
                    (my_lr == max_neighbor_lr and self.agent_id < min(top_agents + [self.agent_id]))
            )
        else:
            self.best_in_group = False

        # שלב 4: הדפסת סטטוס
        print(f"[BEST GROUP] Agent {self.agent_id} | "
              f"LR: {my_lr:.2f} | Max neighbor LR: {max_neighbor_lr:.2f} | Best: {self.best_in_group}")

        # שלב 5: שליחת אישור לפרטנר אם אני מציע
        if self.partner_id is not None:
            self.outbox.append({
                'type': 'partner_approval',
                'approved': self.best_in_group,
                'sender': self.agent_id,
                'to': self.partner_id
            })
            print(
                f"[APPROVAL SENT] Agent {self.agent_id} → {self.partner_id} | I am best_in_group = {self.best_in_group}")

    def phase5_change_value(self):
        if self.problem_type != 'general':
            return

        my_lr = self.shared_lr if self.partner_approved and self.shared_lr is not None else self.compute_local_lr()
        max_neighbor_lr = max(self.lr_from_neighbors.values(), default=0)

        # מקרה של סוכן בודד (MGM רגיל)
        if self.partner_id is None:
            if self.best_in_group and my_lr > 0 and self.proposed_assignment != self.value:
                print(f"[CHANGE SOLO] Agent {self.agent_id}: {self.value} → {self.proposed_assignment} | "
                      f"My LR: {my_lr:.2f} | Max neighbor LR: {max_neighbor_lr:.2f}")
                self.value = self.proposed_assignment

        # מקרה של זוג (MGM-2)
        else:
            if self.best_in_group and self.partner_approved and my_lr > 0 and self.proposed_assignment != self.value:
                print(f"[CHANGE PAIR] Agent {self.agent_id}: {self.value} → {self.proposed_assignment} | "
                      f"My LR: {my_lr:.2f} | Partner approved")
                self.value = self.proposed_assignment

    def compute_local_lr(self):
        # שלב 1: קבלת ערכים של שכנים מתוך value_broadcast בלבד
        neighbor_messages = [msg for msg in self.inbox if msg.get('type') == 'value_broadcast']

        # שלב 2: חישוב עלות נוכחית
        current_cost = self.compute_cost(self.value, neighbor_messages)
        best_value = self.value
        best_cost = current_cost

        # שלב 3: מציאת השמה משופרת (אם קיימת)
        for value in self.domain:
            if value == self.value:
                continue
            cost = self.compute_cost(value, neighbor_messages)
            if cost < best_cost:
                best_value = value
                best_cost = cost

        # שלב 4: שמירה פנימית של החלטה (אופציונלי, כמו ב-MGM)
        self.last_decision = {
            'agent_id': self.agent_id,
            'gain': current_cost - best_cost,
            'new_value': best_value,
            'current_value': self.value
        }

        return current_cost - best_cost

    def _get_neighbor_values(self):
        return [msg for msg in self.inbox if 'value' in msg and msg['sender'] in self.neighbors]


















####################################3

import random
import networkx as nx
import matplotlib.pyplot as plt

class CreateEnvironment:
    def __init__(self, num_agents=30, domain=None, problem_type='general', seed=123):
        self.num_agents = num_agents
        self.problem_type = problem_type
        self.seed = seed
        self.agents = []  # ניצור משתנה לשמירת הסוכנים
        if domain is not None:
            self.domain = domain
        elif self.problem_type == 'coloring':
            self.domain = ['red', 'green', 'blue']  # דומיין קטן לצביעת גרף (3 צבעים)
        else:
            self.domain = ['a', 'b', 'c', 'd', 'e']  # דומיין רגיל

    def create_agents(self):
        random.seed(self.seed)
        self.agents = [
            Agent(agent_id=i, domain=self.domain, problem_type=self.problem_type)
            for i in range(1, self.num_agents + 1)
        ]

        return self.agents

    def connect_agents(self, k, seed=None):
        if seed is not None:
            random.seed(seed)

        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                if random.random() < k:
                    agent_i = self.agents[i]
                    agent_j = self.agents[j]

                    # חיבור סימטרי ברשימת שכנים
                    agent_i.neighbors.append(agent_j.agent_id)
                    agent_j.neighbors.append(agent_i.agent_id)

                    # יצירת טבלת עלויות סימטרית
                    table = {}
                    for val_i in agent_i.domain:
                        for val_j in agent_j.domain:
                            if self.problem_type == 'general':
                                cost = random.randint(100, 200)
                            elif self.problem_type == 'coloring':
                                cost = 0 if val_i != val_j else 100
                            table[(val_i, val_j)] = cost
                            table[(val_j, val_i)] = cost  # סימטרי

                    # הקצאה לשני הסוכנים
                    agent_i.cost_tables[agent_j.agent_id] = table
                    agent_j.cost_tables[agent_i.agent_id] = table


        return self.agents

    def draw_graph(self):
        """מצייר את הגרף של הסוכנים והחיבורים ביניהם."""
        G = nx.Graph()

        # הוספת קודקודים
        for agent in self.agents:
            G.add_node(agent.agent_id)

        # הוספת קשתות
        for agent in self.agents:
            for neighbor_id in agent.neighbors:
                if not G.has_edge(agent.agent_id, neighbor_id):
                    G.add_edge(agent.agent_id, neighbor_id)

        # ציור
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        plt.title("רשת סוכנים - חיבורים בין סוכנים")
        plt.show()

    @staticmethod
    def draw_colored_graph(agents):
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.Graph()

        for agent in agents:
            G.add_node(agent.agent_id)

        for agent in agents:
            for neighbor_id in agent.neighbors:
                if not G.has_edge(agent.agent_id, neighbor_id):
                    G.add_edge(agent.agent_id, neighbor_id)

        color_mapping = {
            'red': 'red',
            'blue': 'blue',
            'green': 'green',

        }

        node_colors = [color_mapping.get(agent.value.lower(), 'gray') for agent in agents]

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500, font_size=10)
        plt.title("גרף צבוע לפי ערכים שנבחרו ע\"י הסוכנים")
        plt.show()



###########################################
class DCOPEnvironment:
    def __init__(self, agents):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.mailboxes = {agent.agent_id: [] for agent in agents}


    def receive_all_messages(self):
        for agent_id, agent in self.agents.items():
            # קבלת ההודעות מהתיבה של הסוכן
            agent.receive_messages(self.mailboxes[agent_id])


    def run_all_agents(self, p, algorithm='DSA',iteration=0):
        """מריץ את כל הסוכנים לפי אלגוריתם נבחר"""
        if algorithm == 'DSA':
            return self._run_dsa_round(p)
        elif algorithm == 'MGM':
            return self._run_mgm_round()
        elif algorithm == 'MGM2':
            return self._run_mgm2_round(iteration)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _run_dsa_round(self, p):
        new_mailboxes = {agent_id: [] for agent_id in self.agents}
        changes = 0

        for agent_id, agent in self.agents.items():
            outgoing_messages, changed = agent.run_dsa(p)
            for msg in outgoing_messages:
                new_mailboxes[msg['receiver']].append(msg)
            if changed:
                changes += 1

        self.mailboxes = new_mailboxes
        return changes

    def _run_mgm_round(self):
        changes = 0

        # 🔹 שלב a: שליחת ערכים נוכחיים לכל השכנים
        value_mailboxes = {agent_id: [] for agent_id in self.agents}
        for agent_id, agent in self.agents.items():
            for neighbor_id in agent.neighbors:
                value_mailboxes[neighbor_id].append({
                    'sender': agent_id,
                    'value': agent.value
                })

        self.mailboxes = value_mailboxes  # ← inbox של כולם עכשיו כולל ערכים מהשכנים

        # 🔹 שלב b–d: חישוב gain לכל סוכן
        gain_messages = {}
        for agent_id, agent in self.agents.items():
            agent.inbox = self.mailboxes[agent_id]
            gain_msg = agent.run_mgm_phase1()
            gain_messages[agent_id] = gain_msg

        # 🔹 שלב e: שליחת gain לשכנים
        gain_mailboxes = {agent_id: [] for agent_id in self.agents}
        for sender_id, gain_msg in gain_messages.items():
            for neighbor_id in self.agents[sender_id].neighbors:
                gain_mailboxes[neighbor_id].append(gain_msg)

        self.mailboxes = gain_mailboxes  # ← inbox מתעדכן עם gain messages

        # 🔹 שלב f+g: החלטה מי מחליף ערך
        for agent_id, agent in self.agents.items():
            agent.inbox = self.mailboxes[agent_id]
            neighbor_gains = agent.inbox
            changed = agent.run_mgm_phase2(neighbor_gains)
            if changed:
                changes += 1

        return changes

    def _run_mgm2_round(self,iteration):
        # שלב 0: שליחת value נוכחי
        agent_last_values = {}

        for agent in self.agents.values():
            for neighbor_id in agent.neighbors:
                self.mailboxes[neighbor_id].append({
                    'type': 'value_broadcast',
                    'value': agent.value,
                    'sender': agent.agent_id
                })
            agent_last_values[agent.agent_id] = agent.value

        num_pairs=0

        # שלב 1: הצעות
        for agent in self.agents.values():
            agent.phase1_propose()
        self._flush_mail()

        # שלב 2: תגובות
        self.receive_all_messages()
        for agent in self.agents.values():
            agent.phase2_respond()
        self._flush_mail()

        # שלב 3: שליחת LR
        self.receive_all_messages()
        for agent in self.agents.values():
            was_in_pair = agent.phase3_send_lr()
            if was_in_pair:
                num_pairs += 1
        self._flush_mail()

        # שלב 4: בדיקת אם אני הכי טוב
        self.receive_all_messages()
        for agent in self.agents.values():
            agent.phase4_check_if_best()
        self._flush_mail()

        # שלב 5: שינוי ערך אם כדאי
        self.receive_all_messages()
        for agent in self.agents.values():
            agent.phase5_change_value()

        print(f"[INFO] Iteration {iteration}: {num_pairs} agents participated in pairs")
        print(f"[INFO]          → {num_pairs // 2} unique pairs formed")

        # חישוב שינויים
        changes = sum(
            int(agent.value != agent_last_values[agent.agent_id])
            for agent in self.agents.values()
        )
        global_cost = self.get_global_cost()
        print(f"[MGM2] Global cost after iteration: {global_cost}")
        return changes

    def _flush_mail(self):
        for agent_id, agent in self.agents.items():
            for msg in agent.outbox:
                recipient = msg['to']
                self.mailboxes[recipient].append({
                    'sender': agent_id,
                    **msg
                })
            agent.outbox = []

    def get_global_cost(self):
        """חישוב עלות כוללת של הרשת"""
        total_cost = 0
        visited_pairs = set()

        for agent_id, agent in self.agents.items():
            for neighbor_id in agent.neighbors:
                if (agent_id, neighbor_id) not in visited_pairs and (neighbor_id, agent_id) not in visited_pairs:
                    my_value = agent.value
                    neighbor_value = self.agents[neighbor_id].value
                    cost = agent.cost_tables[neighbor_id].get((my_value, neighbor_value), 0)
                    total_cost += cost
                    visited_pairs.add((agent_id, neighbor_id))

        return total_cost

    def print_cost_tables(self):
        print("\n--- טבלאות עלויות לכל סוכן מול שכניו ---")
        for agent_id, agent in self.agents.items():
            print(f"\nסוכן {agent_id}:")
            for neighbor_id, table in agent.cost_tables.items():
                print(f"  מול שכן {neighbor_id}:")
                for (my_val, neighbor_val), cost in sorted(table.items()):
                    print(f"    ({my_val}, {neighbor_val}) → {cost}")


###############################################################


class Simulator:
    def __init__(self, environment):
        self.environment = environment
        self.costs_over_time = []

    def run(self, p=1.0, algorithm='DSA', max_iterations=20):
        """
        מריץ את האלגוריתם הנתון עד התכנסות או עד max_iterations.
        תומך ב־DSA, MGM, MGM2.
        אין הדפסות, מיועד להרצות מרובות.
        """
        self.costs_over_time = []

        # ✨ הוספת העלות הרנדומית ההתחלתית (איטרציה 0)
        initial_cost = self.environment.get_global_cost()
        self.costs_over_time.append(initial_cost)

        min_iterations_before_checking_convergence = 5

        for iteration in range(max_iterations):
            self.environment.receive_all_messages()

            changes = self.environment.run_all_agents(p=p, algorithm=algorithm,iteration=iteration)
            cost = self.environment.get_global_cost()
            self.costs_over_time.append(cost)






import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


# פונקציה: להריץ 30 בעיות ולחשב ממוצע על כל איטרציה
def average_costs_over_runs_shared_problems(
    ps=[0.2, 0.7, 1.0],
    k=0.25,
    problem_type='general',
    max_iterations=20,
    num_runs=10,
    algorithms = ['DSA', 'MGM', 'MGM2']
):
    # רק אלגוריתמים שתלויים בפרמטר p
    algorithms_with_p = {'DSA'}

    # מילון התוצאות: alg -> p -> רשימות עלויות
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
                copied_agents = deepcopy_agents(agents)
                env = DCOPEnvironment(copied_agents)
                sim = Simulator(env)

                sim.run(
                    p=p if p is not None else 1.0,
                    algorithm=alg,
                    max_iterations=max_iterations
                )

                results[alg][p].append(sim.costs_over_time)

    # השלמה לריצות קצרות (אם יש פחות איטרציות)
    for alg in results:
        for p in results[alg]:
            for i in range(len(results[alg][p])):
                run_costs = results[alg][p][i]
                expected_length = max_iterations + 1
                if len(run_costs) < expected_length:
                    last_value = run_costs[-1]
                    run_costs += [last_value] * (expected_length - len(run_costs))

    # ממוצע על כל ההרצות
    return {
        alg: {
            p: np.mean(np.array(results[alg][p]), axis=0)
            for p in results[alg]
        }
        for alg in algorithms
    }



def deepcopy_agents(agents_list):
    from copy import deepcopy
    new_agents = []

    id_to_agent = {}

    for agent in agents_list:
        new_agent = Agent(
            agent_id=agent.agent_id,
            domain=deepcopy(agent.domain)
        )
        new_agent.value = agent.value
        new_agent.cost_tables = deepcopy(agent.cost_tables)
        id_to_agent[new_agent.agent_id] = new_agent
        new_agents.append(new_agent)

    # עדכון שכנים לפי מזהים
    for original_agent in agents_list:
        copied_agent = id_to_agent[original_agent.agent_id]
        copied_agent.neighbors = list(original_agent.neighbors)

    return new_agents



# פונקציה: מציירת את הגרף ל-k מסוים
def plot_algorithms_for_k_fixed_problems(k, problem_type='general', save_as=None):
    ps = [0.2, 0.7, 1.0]
    algorithms = ['DSA', 'MGM', 'MGM2']  # הוספנו את MGM2
    max_iterations = 20

    results = average_costs_over_runs_shared_problems(
        ps=ps,
        k=k,
        problem_type=problem_type,
        max_iterations=max_iterations,
        algorithms=algorithms
    )

    plt.figure(figsize=(12, 6))

    for alg in algorithms:
        # אם האלגוריתם תלוי ב־p (כמו DSA), נציג קווים לכל ערך של p
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




# פונקציה ראשית: להריץ הכל
def run_all_dcop_algorithms():
    plot_algorithms_for_k_fixed_problems(k=0.25, problem_type='general')
    plot_algorithms_for_k_fixed_problems(k=0.75, problem_type='general')
    ##plot_algorithms_for_k_fixed_problems(k=0.1, problem_type='coloring')

# הפעלה
run_all_dcop_algorithms()








