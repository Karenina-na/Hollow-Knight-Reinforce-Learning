# -*- coding: utf-8 -*-
import time
from Tool.Actions import take_action, restart, take_direction, Nothing
import collections
from Tool.FrameBuffer import FrameBuffer
from ReplayMemory import ReplayMemory
from Agent import Agent
from Tool.Helper import pause_game, move_judge, action_judge
from Tool.GetHP import Hp_getter

WIDTH = 400
HEIGHT = 200

ACTION_DIM = 8
MOVE_DIM = 4

FRAMEBUFFERSIZE = 4

state_size = (230, 230, 1670, 930)

MEMORY_SIZE = 200  # replay memory size
MEMORY_WARMUP_SIZE = 24  # replay memory pre store some data

MAX_EPISODES = 1000  # max episodes
BATCH_SIZE = 10  # memory batch size
LEARNING_RATE = 0.00001  # learning rate
GAMMA = 0.99  # reward discount
UPDATE_TARGET_FREQUENCY = 20  # update target network frequency
E_GREEDY = 0.12  # greedy policy
E_GREEDY_DECAY = 1e-6  # decay of greedy policy

move_name = ["Move_Left", "Move_Right", "Turn_Left", "Turn_Right"]

action_name = ["Attack", "Attack_Up",
               "Short_Jump", "Mid_Jump", "Skill_Up",
               "Skill_Down", "Rush", "Cure"]
DELAY_REWARD = 1


def one_episode(hp, agent, memory, PASS_COUNT, paused):
    restart(state_size)

    # learn while load game

    for i in range(8):
        if len(memory) > MEMORY_WARMUP_SIZE:
            # print("move and action learning")
            batch_state, batch_moves, batch_actions, batch_reward, batch_next_state, batch_done = memory.sample(
                BATCH_SIZE)
            agent.train_move_action_value(batch_state, batch_moves, batch_actions, batch_reward, batch_next_state,
                                          batch_done, GAMMA)

    step = 0
    done = 0
    total_reward = 0

    start_time = time.time()
    # Delay Reward
    DelayMoveReward = collections.deque(maxlen=DELAY_REWARD)
    DelayActReward = collections.deque(maxlen=DELAY_REWARD)
    DelayStation = collections.deque(maxlen=DELAY_REWARD + 1)  # 1 more for next_station
    DelayActions = collections.deque(maxlen=DELAY_REWARD)
    DelayDirection = collections.deque(maxlen=DELAY_REWARD)

    # enter the game
    while True:
        boss_hp_value = hp.get_boss_hp()
        self_hp = hp.get_self_hp()
        if 800 < boss_hp_value <= 900 and 1 <= self_hp <= 9:
            break

    # System Control
    thread = FrameBuffer(1, "FrameBuffer", WIDTH, HEIGHT, state_size, maxlen=FRAMEBUFFERSIZE)
    thread.start()

    last_hornet_y = 0
    print("start")
    while True:
        step += 1

        while len(thread.buffer) < FRAMEBUFFERSIZE:
            time.sleep(0.1)

        # get state
        state = thread.get_buffer()
        # get HP
        boss_hp_value = hp.get_boss_hp()
        self_hp = hp.get_self_hp()
        # get position
        player_x, player_y = hp.get_play_location()
        hornet_x, hornet_y = hp.get_hornet_location()
        # get soul
        soul = hp.get_souls()

        hornet_skill1 = False
        if 32 < last_hornet_y < 32.5 and 32 < hornet_y < 32.5:
            hornet_skill1 = True
        last_hornet_y = hornet_y

        move, action = agent.sample(state)

        # take action
        take_direction(move)
        take_action(action)

        # get Next State
        next_state = thread.get_buffer()
        # get Next HP
        next_boss_hp_value = hp.get_boss_hp()
        next_self_hp = hp.get_self_hp()
        # get Next position
        next_player_x, next_player_y = hp.get_play_location()
        next_hornet_x, next_hornet_y = hp.get_hornet_location()

        # get reward
        move_reward = move_judge(self_hp, next_self_hp, player_x, next_player_x, hornet_x, next_hornet_x,
                                 move, hornet_skill1)
        # print(move_reward)
        act_reward, done = action_judge(boss_hp_value, next_boss_hp_value, self_hp, next_self_hp,
                                        next_player_x, next_hornet_x, next_hornet_x, action, hornet_skill1)
        # print(reward)
        # print( action_name[action], ", ", move_name[d], ", ", reward)

        DelayMoveReward.append(move_reward)
        DelayActReward.append(act_reward)
        DelayStation.append(state)
        DelayDirection.append(move)
        DelayActions.append(action)

        if len(DelayStation) >= DELAY_REWARD + 1:
            if DelayMoveReward[0] != 0:
                memory.append(
                    (DelayStation[0], DelayDirection[0], DelayActions[0], DelayMoveReward[0], DelayStation[1], done)
                )

        state = next_state
        self_hp = next_self_hp
        boss_hp_value = next_boss_hp_value

        # if (len(act_rmp) > MEMORY_WARMUP_SIZE and int(step/ACTION_SEQ) % LEARN_FREQ == 0):
        #     print("action learning")
        #     batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp.sample(BATCH_SIZE)
        #     algorithm.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)

        total_reward += act_reward + move_reward
        paused = pause_game(paused)

        if done == 1:
            Nothing()
            break
        elif done == 2:
            PASS_COUNT += 1
            Nothing()
            time.sleep(3)
            break

    thread.stop()

    # learning while game over
    for i in range(8):
        if len(memory) > MEMORY_WARMUP_SIZE:
            # print("move and action learning")
            batch_state, batch_moves, batch_actions, batch_reward, batch_next_state, batch_done = memory.sample(
                BATCH_SIZE)
            agent.train_move_action_value(batch_state, batch_moves, batch_actions, batch_reward, batch_next_state,
                                          batch_done, GAMMA)

    return total_reward, step, PASS_COUNT, self_hp


# training
if __name__ == '__main__':
    total_remind_hp = 0

    # experience pool
    memory_pool = ReplayMemory(MEMORY_SIZE, file_name='./Model/memory')

    # agent
    agent = Agent(3, MOVE_DIM, ACTION_DIM, e_greed=E_GREEDY, e_greed_decrement=E_GREEDY_DECAY, lr=LEARNING_RATE,
                  GAMMA=GAMMA, load_path="./Model/agent")

    # Hp counter
    hp = Hp_getter()

    # paused at the beginning
    paused = True
    paused = pause_game(paused)

    # start training
    episode = 0
    PASS_COUNT = 0  # pass count
    while episode < MAX_EPISODES:
        # training
        episode += 1
        if episode % UPDATE_TARGET_FREQUENCY == 1:
            agent.update_target()

        total_reward, total_step, PASS_COUNT, remind_hp = one_episode(hp, agent, memory_pool, PASS_COUNT, paused)
        if episode % 10 == 1:
            agent.save("./Model/agent")
        if episode % 5 == 0:
            memory_pool.save("./Model/memory")
        total_remind_hp += remind_hp
        print("Episode: ", episode, ", pass_count: ", PASS_COUNT, ", hp:", total_remind_hp / episode)
