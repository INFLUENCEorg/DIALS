import time
import numpy as np
import multiprocessing as mp
import multiprocessing.connection

def train_agent(agent, env, training_steps):
  
    obs = env.reset()
    step = 0
    done = False

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

            if done:
                obs = env.reset()
        
        agent.bootstrap(obs)

        if agent.buffer.is_full:
            agent.update()
    
    return agent

def local_trainer_process(remote: multiprocessing.connection.Connection, agent, env):

    while True:
        cmd, arg = remote.recv()

        if cmd == 'train_influence':
            inf_loss = env.influence.learn()
            remote.send(inf_loss)

        elif cmd == 'train_agent':
            agent = train_agent(agent, env, arg)
            remote.send(agent)

        elif cmd == 'close':
            env.close()
            remote.close()
            break

        else:
            raise NotImplementedError

class LocalTrainer(object):

    def __init__(self, agent, env, training_steps):
        self.child, parent = mp.Pipe()
        self.process = mp.Process(target=local_trainer_process, args=(parent, agent, env))
        self.process.start()
