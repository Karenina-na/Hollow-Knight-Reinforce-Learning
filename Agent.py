import numpy as np
from Net import Policy_move, Policy_action, Value
import torch
import torch.nn.functional as F
import os


class Agent:
    def __init__(self, state_dim, move_num, act_num, e_greed=0.1, e_greed_decrement=0, lr=0.001, GAMMA=0.9,
                 load_path=None):
        # local and target network
        self.policy_move_local = Policy_move(state_dim, move_num)
        self.policy_move_target = Policy_move(state_dim, move_num)

        self.policy_action_local = Policy_action(state_dim, act_num)
        self.policy_action_target = Policy_action(state_dim, act_num)

        self.value_local = Value(state_dim)
        self.value_target = Value(state_dim)

        # variables
        self.move_num = move_num
        self.act_dim = act_num
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement
        self.learning_rate = lr
        self.GAMMA = GAMMA
        self.load_path = load_path

        # optimizer
        self.optimizer_move = torch.optim.Adam(self.policy_move_local.parameters(), lr=self.learning_rate)
        self.optimizer_action = torch.optim.Adam(self.policy_action_local.parameters(), lr=self.learning_rate)
        self.optimizer_value = torch.optim.Adam(self.value_local.parameters(), lr=self.learning_rate)

        # distribution
        self.distribution = torch.distributions.Categorical

        # load
        self.load(load_path)

    def sample(self, state, soul, hornet_x, hornet_y, player_x, hornet_skill1):
        """
        Sample an action from the agent's policy.
        :param state: the current state of the environment
        :param soul:    the current soul of the hornet
        :param hornet_x:    the current x of the hornet
        :param hornet_y:    the current y of the hornet
        :param player_x:    the current x of the player
        :param hornet_skill1:   the current skill1 of the hornet
        :return:    the action and move sampled from the agent's policy
        """
        # predict the action and move
        pred_move = self.policy_move_local.choose_move(state)
        pred_act = self.policy_action_local.choose_action(state)

        # random move
        sample = np.random.rand()
        move = pred_move if sample > self.e_greed else self.__better_move(hornet_x, player_x, hornet_skill1)

        # change the e_greed
        self.e_greed = max(
            0.03, self.e_greed - self.e_greed_decrement)

        # random action
        sample = np.random.rand()
        act = pred_act if sample > self.e_greed else self.__better_action(soul, hornet_x, hornet_y, player_x, hornet_skill1)

        # change the e_greed
        self.e_greed = max(
            0.03, self.e_greed - self.e_greed_decrement)

        return move, act

    def __better_action(self, soul, hornet_x, hornet_y, player_x, hornet_skill1):
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

    def __better_move(self, hornet_x, player_x, hornet_skill1):
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

    def update_target(self):
        """
        Update the target network.
        """
        self.policy_move_target.load_state_dict(self.policy_move_local.state_dict())
        self.policy_action_target.load_state_dict(self.policy_action_local.state_dict())
        self.value_target.load_state_dict(self.value_local.state_dict())

    def value_state(self, state):
        """
        Calculate the value of a state.
        :param state: the state to be calculated
        :return: the value of the state
        """
        return self.value_local.forward(state)

    def train_move_action_value(self, state, moves, actions, rewards, next_state, done, gamma):
        """
        Train the agent.
        :param state:   the state of the environment
        :param moves:   the moves sampled from the agent's policy
        :param actions: the actions sampled from the agent's policy
        :param rewards: the rewards of the environment
        :param next_state:  the next state of the environment
        :param done:    the done of the environment
        :param gamma:   the discount factor
        :return:    the loss of the agent
        """
        next_state_target = self.value_local.forward(next_state)
        target = rewards + gamma * next_state_target
        if done:
            target = rewards

        # calculate the TD error
        value = self.value_local.forward(state)
        td_error = target - value

        # update the value network
        self.optimizer_value.zero_grad()
        value_loss = td_error.pow(2)
        value_loss.backward()
        self.optimizer_value.step()

        # calculate the advantage
        advantage = td_error.detach()

        # calculate the loss of the move network
        move_pred = self.policy_move_local.forward(state)
        move_log_prob = F.softmax(move_pred, dim=1)
        move_distribution = self.distribution(move_log_prob)
        move_loss = -move_distribution.log_prob(moves) * advantage.detach().squeeze()

        # calculate the loss of the action network
        action_pred = self.policy_action_local.forward(state)
        action_log_prob = F.softmax(action_pred, dim=1)
        action_distribution = self.distribution(action_log_prob)
        action_loss = -action_distribution.log_prob(actions) * advantage.detach().squeeze()

        # update the move and action network
        self.optimizer_move.zero_grad()
        self.optimizer_action.zero_grad()
        move_loss.backward()
        action_loss.backward()
        self.optimizer_move.step()
        self.optimizer_action.step()

        # return loss
        return move_loss, action_loss, value_loss

    def save(self, path):
        """
        Save the agent.
        :param path:    the path to save the agent
        """
        torch.save(self.policy_move_local.state_dict(), path + '-policy_move_local.pth')
        torch.save(self.policy_move_target.state_dict(), path + '-policy_move_target.pth')
        torch.save(self.policy_action_local.state_dict(), path + '-policy_action_local.pth')
        torch.save(self.policy_action_target.state_dict(), path + '-policy_action_target.pth')
        torch.save(self.value_local.state_dict(), path + '-value_local.pth')
        torch.save(self.value_target.state_dict(), path + '-value_target.pth')

    def load(self, path):
        """
        Load the agent.
        :param path:    the path to load the agent
        """
        if path is None:
            return
        if os.path.exists(path + '-policy_move_local.pth') and os.path.exists(path + '-policy_move_target.pth') and \
                os.path.exists(path + '-policy_action_local.pth') and os.path.exists(path + '-policy_action_target.pth') and \
                os.path.exists(path + '-value_local.pth') and os.path.exists(path + '-value_target.pth'):
            self.policy_move_local.load_state_dict(torch.load(path + '-policy_move_local.pth'))
            self.policy_move_target.load_state_dict(torch.load(path + '-policy_move_target.pth'))
            self.policy_action_local.load_state_dict(torch.load(path + '-policy_action_local.pth'))
            self.policy_action_target.load_state_dict(torch.load(path + '-policy_action_target.pth'))
            self.value_local.load_state_dict(torch.load(path + '-value_local.pth'))
            self.value_target.load_state_dict(torch.load(path + '-value_target.pth'))


if __name__ == '__main__':
    states = torch.randn(1, 3, 800, 800)

    # state_dim, move_num, action_num, e_greed, e_greed_decrement, lr, GAMMA, load_path
    agent = Agent(3, 2, 4, e_greed=0.1, e_greed_decrement=0, lr=0.001, GAMMA=0.9, load_path=None)

    # state, soul, hornet_x, hornet_y, player_x, hornet_skill1
    move, action = agent.sample(states, 5, 10, 10, 0, False)
    print(move, action)

    # state
    value = agent.value_state(states)
    print(value)

    # states, moves, actions, rewards, next_state, done, gamma
    move = torch.tensor(move)
    action = torch.tensor(action)
    next_state = torch.randn(1, 3, 800, 800)
    move_loss, action_loss, value_loss = agent.train_move_action_value(states, move, action, 1, next_state, False, 0.9)
    print(move_loss, action_loss, value_loss)

    # update the target network
    agent.update_target()

