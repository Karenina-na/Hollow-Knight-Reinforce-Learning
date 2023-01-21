import numpy as np
from Net import Net
import torch


class Agent:
    def __init__(self, s_dim, move_num, act_num, e_greed=0.1, e_greed_decrement=0, lr=0.001, load_path=None):
        self.move_num = move_num
        self.act_dim = act_num
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement
        self.model = Net(s_dim, move_num, act_num, lr)

    def sample(self, station, soul, hornet_x, hornet_y, player_x, hornet_skill1):
        """
        Sample an action from the agent's policy.
        :param station: the current state of the environment
        :param soul:    the current soul of the hornet
        :param hornet_x:    the current x of the hornet
        :param hornet_y:    the current y of the hornet
        :param player_x:    the current x of the player
        :param hornet_skill1:   the current skill1 of the hornet
        :return:    the action and move sampled from the agent's policy
        """
        # predict the action and move
        pred_move, pred_act = self.model.predict(station)

        # random move
        sample = np.random.rand()
        if sample < self.e_greed:
            move = self.better_move(hornet_x, player_x, hornet_skill1)
        else:
            move = np.argmax(pred_move)

        # change the e_greed
        self.e_greed = max(
            0.03, self.e_greed - self.e_greed_decrement)

        # random action
        sample = np.random.rand()
        if sample < self.e_greed:
            act = self.better_action(soul, hornet_x, hornet_y, player_x, hornet_skill1)
        else:
            act = np.argmax(pred_act)
            if soul < 33:
                if act == 4 or act == 5:
                    pred_act[0][4] = -30
                    pred_act[0][5] = -30
            act = np.argmax(pred_act)

        # change the e_greed
        self.e_greed = max(
            0.03, self.e_greed - self.e_greed_decrement)
        return move, act

    def better_action(self, soul, hornet_x, hornet_y, player_x, hornet_skill1):
        """
        Sample an action from the agent's policy.
        :param soul:    the current soul of the hornet
        :param hornet_x:    the current x of the hornet
        :param hornet_y:    the current y of the hornet
        :param player_x:    the current x of the player
        :param hornet_skill1:   the current skill1 of the hornet
        :return:    the action sampled from the agent's policy
        """
        dis = abs(player_x - hornet_x)
        if hornet_skill1:
            if dis < 3:
                return 6
            else:
                return 1

        if hornet_y > 34 and dis < 5 and soul >= 33:
            return 4

        if dis < 1.5:
            return 6
        elif dis < 5:
            if hornet_y > 32:
                return 6
            else:
                act = np.random.randint(self.act_dim)
                if soul < 33:
                    while act == 4 or act == 5:
                        act = np.random.randint(self.act_dim)
                return act
        elif dis < 12:
            act = np.random.randint(2)
            return 2 + act
        else:
            return 6

    def better_move(self, hornet_x, player_x, hornet_skill1):
        """
        Sample a move from the agent's policy.
        :param hornet_x:    the current x of the hornet
        :param player_x:    the current x of the player
        :param hornet_skill1:   the current skill1 of the hornet
        :return:    the move sampled from the agent's policy
        """
        # distance between hornet and player
        dis = abs(player_x - hornet_x)
        dire = player_x - hornet_x

        # skill
        if hornet_skill1:
            # run away while distance < 6
            if dis < 6:
                if dire > 0:
                    return 1
                else:
                    return 0
            # do not do long move while distance > 6
            else:
                if dire > 0:
                    return 2
                else:
                    return 3

        # distance control
        if dis < 2.5:
            if dire > 0:
                return 1
            else:
                return 0
        elif dis < 5:
            if dire > 0:
                return 2
            else:
                return 3
        else:
            if dire > 0:
                return 0
            else:
                return 1

    def train_move_action(self, state, moves, actions, rewards, next_state, done, gamma=0.99):
        """
        Train the agent.
        :param state:   the state of the environment
        :param action:  the action of the agent
        :param move:    the move of the agent
        """
        _, _, value = self.model.forward(state)
        target = rewards + gamma * value
        if done:
            target = rewards

        # calculate the loss
        loss = self.model.loss_func(state, moves, actions, target)

        # backward
        self.model.optimizer.zero_grad()
        loss.backward()

    def get_dict(self):
        """
        Get the dictionary of the agent.
        :return:    the dictionary of the agent
        """
        return self.model.state_dict()

    def save(self, path):
        """
        Save the agent.
        :param path:    the path to save the agent
        """
        self.model.save(path)


if __name__ == '__main__':
    s = torch.randn(1, 3, 800, 800)
    A = Agent(3, 2, 3, e_greed=0.5)
    for i in range(0, 10):
        s = torch.randn(1, 3, 800, 800)
        print(A.sample(s, 10, 5, 1, 0, False))
