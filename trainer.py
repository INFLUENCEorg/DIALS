import time
import numpy as np
import multiprocessing as mp
import multiprocessing.connection
from torch.multiprocessing import Pool, Process, set_start_method

def train_single_agent(agent, env, training_steps=1.0e+4):
  
    obs = env.reset()
    step = 0
    done = [True]
    while step <= training_steps:
        
        rollout_step = 0
        while rollout_step < agent.rollout_steps:

            if agent.policy.recurrent:
                agent.reset_hidden_memory(done)
                hidden_memory = agent.policy.hidden_memory
            else:
                hidden_memory = None
            action, value, log_prob = agent.choose_action(obs)
            new_obs, reward, done, _ = env.step(action)
            agent.add_to_memory(obs, action, reward, done, value, log_prob, hidden_memory)
            obs = new_obs
            rollout_step += 1
            step += 1

            if done[0]:
                obs = env.reset()
        agent.bootstrap(obs)

        if agent.buffer.is_full:
            agent.update()
    return agent

def train_multi_agent(agents, env, training_steps=1.0e+3):
  
    obs = env.reset()
    step = 0
    done = [True]*len(agents)
    while step <= training_steps:
        
        rollout_step = 0
        while rollout_step < agents[0].rollout_steps:
            
            hidden_memories = []
            actions = []
            values = []
            log_probs = []
            for i, agent in enumerate(agents):
                if agent.policy.recurrent:
                    agent.reset_hidden_memory([done[i]])
                    hidden_memories.append(agent.policy.hidden_memory)
                else:
                    hidden_memories.append(None)
                action, value, log_prob = agent.choose_action(obs[i])
                actions.append(action)
                values.append(value)
                log_probs.append(log_prob)

            new_obs, reward, done, _ = env.step(actions)

            for i, agent in enumerate(agents):
                agent.add_to_memory(obs[i], actions[i], [reward[i]], [done[i]], values[i], log_probs[i], hidden_memories[i])
            
            obs = new_obs
            rollout_step += 1
            step += 1

            if done[0]:
                obs = env.reset()

        for agent in agents:
            agent.bootstrap(obs[i])
            if agent.buffer.is_full:
                agent.update()

    return agents


class DistributedTrainer(object):
    
    def __init__(self, agents, envs):
        # self.local_trainers = [LocalTrainer(agent, env) for agent, env, in zip(agents, envs)]
        # self.pool =  Pool(processes=len(agents))
        self.agents = agents
        self.envs = envs
        # set_start_method('fork')
            

    def train(self, training_steps):

        agents = []
        # for local_trainer in self.local_trainers:
        #     local_trainer.child.send(('train_agent', training_steps))
        #     agents.append(local_trainer.child.recv())
        # pool = Pool()#processes=len(self.agents))
        # outputs = []
        # for i in range(len(self.agents)):
        #     outputs.append(pool.apply_async(train_single_agent, (self.agents[i], self.envs[i], training_steps)))

        # agents = [output.get() for output in outputs]
        # pool.close()
        # processes = []
        # for agent, env in zip(self.agents, self.envs):
        #     p = Process(target=train_single_agent, args=(agent, env, training_steps))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
        with Pool() as pool:
            self.agents = pool.starmap(train_single_agent, zip(self.agents, self.envs))
        return self.agents

    def train_influence(self):

        # for local_trainer in self.local_trainers:
        #     local_trainer.child.send(('train_influence', None))
        # pool = Pool()#processes=len(self.agents))
        # for i in range(len(self.agents)):
        #     pool.apply_async(self.envs[i].influence.learn, ())
        # pool.close()
        # pool.join()
        processes = []
        for env in self.envs:
            p = Process(target=env.influence.learn)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def close(self):

        for env in self.envs:
            env.close()
    

class GlobalTrainer(object):

    def __init__(self, agents, env):

        self.agents = agents
        self.env = env
    
    def train(self, training_steps):

        agents = train_multi_agent(self.agents, self.env, training_steps)
        return agents

    def close(self):

        self.env.close()
