import random
from operator import truediv

import networkx as nx
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, agent_id, domain, problem_type,environment):
        self.agent_id = agent_id
        self.domain = domain[:]
        self.problem_type = problem_type
        self.value = random.choice(self.domain)
        self.neighbors = []
        self.cost_tables = {}
        self.inbox = []
        self.environment = environment
        self.partner_proposal_msg=None

        # MGM-2
        self.is_proposer = False
        self.partner_id = None
        self.proposed_assignment = None #the best value for each
        self.shared_lr = None
        self.partner_approved = False
        self.received_proposals = []
        self.lr_from_neighbors = {}
        self.outbox = []
        self.in_partnership  =False
        self.lr_to_send=None
        self.partner_proposal_msg = None

    #reset before mgm2 itartion rest cuples
    def reset_for_iteration(self):
        self.is_proposer = False
        self.in_partnership  = False
        self.partner_id = None
        self.partner_proposal_msg = None
        self.shared_lr = None
        self.proposed_assignment = self.value
        self.lr_to_send = None
        self.best_in_group=False


    def receive_messages(self, messages):
        self.inbox = messages

    def compute_cost(self, my_value, received_messages):
        """חישוב עלות נוכחית או עבור ערך אלטרנטיבי."""
        total_cost = 0
        for msg in received_messages:
            if 'value' not in msg:
                continue

            neighbor_id = msg['sender']
            neighbor_value = msg['value']
            cost_table = self.cost_tables.get(neighbor_id)
            if cost_table is not None:
                cost = cost_table.get((my_value, neighbor_value), 0)
                total_cost += cost
            else:

                continue
        return total_cost

    def run_mgm_phase1(self):
        # calc the best assimgent and make messege with the LR
        value_messages = [msg for msg in self.inbox if 'value' in msg]


        current_cost = self.compute_cost(self.value, value_messages)


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

    @staticmethod
    def compute_best_joint_assignment(
            self_value, self_domain, self_neighbors, self_cost_tables,proposer_id,
            proposer_value, proposer_domain, proposer_neighbors, proposer_cost_tables,
            neighbor_values
    ):


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
            partner_choice = random.choice(self.neighbors)
            message = {
                'type': 'proposal',
                'value': self.value,
                'domain': self.domain,
                'neighbor_values': {neighbor: self.environment.agents[neighbor].value for neighbor in self.neighbors},
                'cost_tables': {
                    neighbor: self.cost_tables.get(neighbor, {}) for neighbor in self.neighbors
                },
                'sender': self.agent_id,
                'to': partner_choice
            }
            self.outbox.append(message)

    def phase2_respond(self):
        if self.is_proposer:
            return False

        # סינון הצעות שהתקבלו
        proposals = [msg for msg in self.inbox if msg['type'] == 'proposal']
        if not proposals:
            return False

        selected = random.choice(proposals)
        proposer_id = selected['sender']
        self.in_partnership = True
        self.partner_id = proposer_id
        neighbor_values = {
            msg['sender']: msg['value']
            for msg in self.inbox
            if msg.get('type') == 'value_broadcast'
        }

        result = self.compute_best_joint_assignment(
            self_value=self.value,
            self_domain=self.domain,
            self_neighbors=self.neighbors,
            self_cost_tables=self.cost_tables,
            proposer_id=proposer_id,
            proposer_value=selected['value'],
            proposer_domain=selected['domain'],
            proposer_neighbors=selected['neighbor_values'],
            proposer_cost_tables=selected['cost_tables'],
            neighbor_values=neighbor_values
        )



        self.partner_proposal_msg = selected
        self.shared_lr = result['joint_lr']
        self.proposed_assignment = result['best_self_value']
        self.lr_to_send = result['joint_lr']

        self.outbox.append({
            'type': 'proposal_response',
            'to': proposer_id,
            'joint_lr': result['joint_lr'],
            'best_partner_assignment': result['best_proposer_value'],
            'from': self.agent_id,
        })

        return True

    def phase3_send_lr(self):
        if self.is_proposer:
            response = next(
                (msg for msg in self.inbox if msg['type'] == 'proposal_response'),
                None  # ברירת מחדל במקרה שאין התוצאה
            )

            if response and 'joint_lr' in response:
                # יש LR מחושב מהפרטנר
                self.in_partnership = True
                self.shared_lr = response['joint_lr']
                self.partner_id = response.get('from')  # אם יש 'from' או תשאיר את הקיים
                self.proposed_assignment = response.get('best_partner_assignment')
                lr = self.shared_lr
                print(
                    f"[PAIR] Agent {self.agent_id} ↔ {response['sender']} | Sender = {self.agent_id} | Joint LR = {lr:.2f}")
            else:
                # לא התקבלה תגובה או אין LR → מחשב LR בעצמי ומאפס שידוך
                lr, best_val = self.compute_local_lr()
                self.proposed_assignment = best_val
                print(f"[SOLO] Agent {self.agent_id} (Proposer, no response) | LR = {lr:.2f}")

        else:
            # מקבל ההצעה
            if self.in_partnership and self.shared_lr is not None:
                lr = self.shared_lr
                print(
                    f"[PAIR] Agent {self.agent_id} ↔ {self.partner_id} | Sender = {self.partner_id} | Joint LR = {lr:.2f}")
            else:
                # במקרה שלא בזוג או אין LR משותף - מחשב LR לבד
                lr, best_val = self.compute_local_lr()
                self.proposed_assignment = best_val
                print(f"[SOLO] Agent {self.agent_id} (Receiver, no partnership) | LR = {lr:.2f}")

        self.lr_to_send = lr
        for neighbor in self.neighbors:
            self.outbox.append({
                'type': 'lr_notification',
                'lr': lr,
                'sender': self.agent_id,
                'to': neighbor
            })

    def phase4_check_if_best(self):
        # שלב 1: איסוף הודעות LR מהשכנים
        lrs = [msg for msg in self.inbox if msg['type'] == 'lr_notification']
        self.lr_from_neighbors = {msg['sender']: msg['lr'] for msg in lrs}



        # שלב 3: מקסימום LR בשכונה כולל עצמי
        all_lrs = self.lr_from_neighbors.copy()
        all_lrs[self.agent_id] = self.lr_to_send

        max_lr = max(all_lrs.values())
        top_agents = [aid for aid, lr in all_lrs.items() if lr == max_lr]
        min_top_agent = min(top_agents)

        self.best_in_group = (self.lr_to_send == max_lr and self.agent_id == min_top_agent)

        ##print(
            ##f"[BEST LR CHECK] Agent {self.agent_id} | My LR = {my_lr:.2f} | Max LR = {max_lr:.2f} | Best: {self.best_in_group}")
       ## print(
            ##f"[LR FROM NEIGHBORS] Agent {self.agent_id} | Neighbors: {self.neighbors} | Received LRs: {self.lr_from_neighbors}")

        # שלב 5: אם יש פרטנר, שלח לו אישור האם אני הכי טוב
        if self.in_partnership:
            self.outbox.append({
                'type': 'partner_approval',
                'approved': self.best_in_group,
                'to': self.partner_id
            })
          ##  print(f"[APPROVAL SENT] Agent {self.agent_id} → {self.partner_id} | Approved = {self.best_in_group}")

    def phase5_change_value(self):
        # חישוב LR (אם בזוג → משותף, אחרת → לוקאלי)


        # קלט: האם השותף שלי חושב שהוא הכי טוב
        self.partner_best = False
        for msg in self.inbox:
            if msg.get('type') == 'partner_approval' and msg.get('sender') == self.partner_id:
                self.partner_best = msg.get('approved', False)

        # 🔹 שינוי יחיד (MGM רגיל)
        if not self.in_partnership:
            if self.best_in_group and self.lr_to_send > 0 and self.proposed_assignment != self.value:
              ##  print(f"[CHANGE SOLO] Agent {self.agent_id}: {self.value} → {self.proposed_assignment} | "
                ##      f"My LR: {my_lr:.2f}")
                self.value = self.proposed_assignment

        # 🔹 שינוי זוגי (MGM-2)
        if (self.in_partnership and
                self.best_in_group and
                self.partner_best and
                my_lr > 0 and
                self.proposed_assignment != self.value):
          ##  print(f"[CHANGE PAIR] Agent {self.agent_id} ⇄ Agent {self.partner_id} | "
            ##      f"{self.value} → {self.proposed_assignment} | "
              ##    f"My LR: {my_lr:.2f}, Partner Best: {self.partner_best}")
            self.value = self.proposed_assignment

        # DEBUG לכל מי שהוא best
       ## if self.best_in_group:
           ## print(
              ##  f"[DEBUG] Agent {self.agent_id} | value = {self.value} | proposed = {self.proposed_assignment} | in_pair = {self.in_partntership} | partner_best = {self.partner_best} | LR = {my_lr}")

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


        return current_cost - best_cost,best_value

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
        self.agents = {}  # ניצור משתנה לשמירת הסוכנים
        if domain is not None:
            self.domain = domain
        elif self.problem_type == 'coloring':
            self.domain = ['red', 'green', 'blue']
        else:
            self.domain = ['a', 'b', 'c', 'd', 'e']

    def create_agents(self):
        random.seed(self.seed)
        self.agents = {
            i: Agent(agent_id=i, domain=self.domain, problem_type=self.problem_type, environment=self)
            for i in range(1, self.num_agents + 1)
        }

        return self.agents

    def connect_agents(self, k, seed=None):
        if seed is not None:
            random.seed(seed)

        for i in self.agents:
            for j in self.agents:
                if random.random() < k:
                    agent_i = self.agents[i]
                    agent_j = self.agents[j]

                    # add a line bothe of the sides
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



###########################################
class DCOPEnvironment:
    def __init__(self, agents_dict):
        self.agents = agents_dict
        self.mailboxes = {agent_id: [] for agent_id in self.agents}



    def receive_all_messages(self):
        for agent_id, agent in self.agents.items():
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

        self.clear_mailboxes()
        for agent in self.agents.values():
            agent.reset_for_iteration()
        for agent in self.agents.values():
            for neighbor_id in agent.neighbors:
                self.mailboxes[neighbor_id].append({
                    'type': 'value_broadcast',
                    'value': agent.value,
                    'sender': agent.agent_id
                })
        self._flush_mail()
        self.receive_all_messages()


        # שלב 1: הצעות
        for agent in self.agents.values():
            agent.phase1_propose()
        self._flush_mail()


        self.receive_all_messages()

        for agent in self.agents.values():
            agent.phase2_respond()
        self._flush_mail()

        # שלב 3: שליחת LR
        self.receive_all_messages()
        for agent in self.agents.values():
             agent.phase3_send_lr()
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


        # חישוב שינויים

    def send_all_lrs(self):
        for agent in self.agents.values():
            lr = agent.lr_to_send
            for neighbor in agent.neighbors:
                self.mailboxes[neighbor].append({
                    'type': 'lr_notification',
                    'lr': lr,
                    'sender': agent.agent_id,
                    'to': neighbor
                })

    def clear_mailboxes(self):
        for agent_id in self.mailboxes:
            self.mailboxes[agent_id] = []

    def assign_random_partners(self):
        assigned = set()

        for agent in self.agents.values():
            if agent.is_proposer or agent.agent_id in assigned:
                continue

            proposals = [msg for msg in agent.inbox if msg['type'] == 'proposal']
            if not proposals:
                continue

            selected = random.choice(proposals)
            proposer_id = selected['sender']

            if proposer_id == agent.agent_id or proposer_id in assigned:
                continue

            # רק מקבל ההצעה מאשר שידוך
            agent.in_partnership= True
            agent.partner_id = proposer_id
            agent.partner_proposal_msg = selected

            assigned.add(agent.agent_id)
            # אל תעדכן את ה-proposer כאן!

            print(f"[MATCHED] Agent {agent.agent_id} ↔ Agent {proposer_id}")

    def _flush_mail(self):
        for agent_id, agent in self.agents.items():
            for msg in agent.outbox:
                recipient = msg['to']
                self.mailboxes[recipient].append({
                    **msg,
                    'sender': agent_id
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

    def wait_until_all_received_neighbor_values(self):
        """
        ממתינה עד שכל הסוכנים קיבלו value_broadcast מכל השכנים שלהם.
        מבצעת receive_all_messages בלולאה עד שהשלב הושלם.
        """
        while True:
            self.receive_all_messages()
            all_received = True

            for agent in self.agents.values():
                received = {msg['sender'] for msg in agent.inbox if msg['type'] == 'value_broadcast'}
                expected = set(agent.neighbors)
                missing = expected - received

                if missing:
                    all_received = False
                    # לוג שימושי למעקב
                    print(f"[WAITING] Agent {agent.agent_id} missing values from: {missing}")

            if all_received:
                break

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
        self.costs_over_time = []

        initial_cost = self.environment.get_global_cost()
        self.costs_over_time.append(initial_cost)


        for iteration in range(max_iterations):
            self.environment.receive_all_messages()

            changes = self.environment.run_all_agents(p=p, algorithm=algorithm,iteration=iteration)
            cost = self.environment.get_global_cost()
            self.costs_over_time.append(cost)














