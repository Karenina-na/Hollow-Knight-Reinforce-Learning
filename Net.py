import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(torch.nn.Module):
    """
    ResBlock_2conv_2d
    """

    def __init__(self, In_channel, Med_channel, Out_channel, DownSample=False):
        super(ResBlock, self).__init__()
        self.stride = 1
        if DownSample:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(In_channel, Med_channel, 3, self.stride, padding=1),
            torch.nn.BatchNorm2d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Med_channel, Out_channel, 3, padding=1),
            torch.nn.BatchNorm2d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv2d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual


def set_init(layer):
    """
    initialize weights
    :param layer:   layer
    :return:    None
    """
    nn.init.normal_(layer.weight, mean=0., std=0.1)
    nn.init.constant_(layer.bias, 0.)


class Policy_move(nn.Module):
    def __init__(self, s_dim, m_num):
        super(Policy_move, self).__init__()
        self.s_dim = s_dim  # state dimension
        self.m_num = m_num  # move number
        self.distribution = torch.distributions.Categorical

        # ResNet-18
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(s_dim, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool2d(3, 2, 1),

            ResBlock(64, 64, 64, False),
            ResBlock(64, 64, 64, False),
            ResBlock(64, 64, 64, False),
            #
            ResBlock(64, 128, 128, True),
            ResBlock(128, 128, 128, False),
            ResBlock(128, 128, 128, False),
            ResBlock(128, 128, 128, False),
            #
            ResBlock(128, 256, 256, True),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            #
            ResBlock(256, 512, 512, True),
            ResBlock(512, 512, 512, False),
            ResBlock(512, 512, 512, False),

            torch.nn.AdaptiveAvgPool2d(1)
        )

        # policy-move
        self.policy_m1 = nn.Linear(512, 512)
        self.policy_m2 = nn.Tanh()
        self.policy_m3 = nn.Linear(512, m_num)

    def forward(self, x):
        feature = self.features(x)

        # flatten
        feature = feature.view(feature.size(0), -1)

        # policy-move
        moves = self.policy_m3(self.policy_m2(self.policy_m1(feature)))

        return moves

    def choose_move(self, state):
        """
        choose move
        :param state:   state
        :return:   move
        """
        self.eval()
        move = self.forward(state)
        # the probability of the move
        prob = F.softmax(move, dim=1).data
        # calculate distribution of the move
        m = self.distribution(prob)
        return m.sample().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Policy_action(nn.Module):
    def __init__(self, s_dim, a_num):
        super(Policy_action, self).__init__()
        self.s_dim = s_dim  # state dimension
        self.a_num = a_num  # action number
        self.distribution = torch.distributions.Categorical

        # ResNet-18
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(s_dim, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool2d(3, 2, 1),

            ResBlock(64, 64, 64, False),
            ResBlock(64, 64, 64, False),
            ResBlock(64, 64, 64, False),
            #
            ResBlock(64, 128, 128, True),
            ResBlock(128, 128, 128, False),
            ResBlock(128, 128, 128, False),
            ResBlock(128, 128, 128, False),
            #
            ResBlock(128, 256, 256, True),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            #
            ResBlock(256, 512, 512, True),
            ResBlock(512, 512, 512, False),
            ResBlock(512, 512, 512, False),

            torch.nn.AdaptiveAvgPool2d(1)
        )

        # policy-action
        self.policy_a1 = nn.Linear(512, 512)
        self.policy_a2 = nn.Tanh()
        self.policy_a3 = nn.Linear(512, a_num)

    def forward(self, x):
        feature = self.features(x)

        # flatten
        feature = feature.view(feature.size(0), -1)

        # policy-action
        actions = self.policy_a3(self.policy_a2(self.policy_a1(feature)))

        return actions

    def choose_action(self, state):
        """
        choose action
        :param state:   state
        :return:   action
        """
        self.eval()
        actions = self.forward(state)
        # the probability of the action
        prob = F.softmax(actions, dim=1).data
        # calculate distribution of the action
        a = self.distribution(prob)
        return a.sample().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Value(nn.Module):
    def __init__(self, s_dim):
        super(Value, self).__init__()
        self.s_dim = s_dim  # state dimension
        self.distribution = torch.distributions.Categorical

        # ResNet-18
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(s_dim, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool2d(3, 2, 1),

            ResBlock(64, 64, 64, False),
            ResBlock(64, 64, 64, False),
            ResBlock(64, 64, 64, False),
            #
            ResBlock(64, 128, 128, True),
            ResBlock(128, 128, 128, False),
            ResBlock(128, 128, 128, False),
            ResBlock(128, 128, 128, False),
            #
            ResBlock(128, 256, 256, True),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            #
            ResBlock(256, 512, 512, True),
            ResBlock(512, 512, 512, False),
            ResBlock(512, 512, 512, False),

            torch.nn.AdaptiveAvgPool2d(1)
        )

        # critic
        self.value_1 = nn.Linear(512, 512)
        self.value_2 = nn.Tanh()
        self.value_3 = nn.Linear(512, 1)

    def forward(self, x):
        feature = self.features(x)

        # flatten
        feature = feature.view(feature.size(0), -1)

        # value
        value = self.value_3(self.value_2(self.value_1(feature)))

        return value

    def predict(self, state):
        """
        predict value
        :param state:   state
        :return:   value
        """
        self.eval()
        value = self.forward(state)
        return value

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == '__main__':
    action_num = 3
    move_num = 2
    state_dim = 3
    s = torch.randn(2, state_dim, 800, 800)
    m = torch.randint(0, move_num, [2, 1])
    a = torch.randint(0, action_num, [2, 1])
    policy_move = Policy_move(state_dim, move_num)
    policy_action = Policy_action(state_dim, action_num)
    value = Value(state_dim)
    print(policy_move(s))
    print(policy_action(s))
    print(value(s))
    print(policy_move.choose_move(s))
    print(policy_action.choose_action(s))
    print(value.predict(s))