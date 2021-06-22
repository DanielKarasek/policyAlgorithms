import tensorflow as tf
import tensorflow.contrib.layers as layers
import gym
from gym.spaces import Discrete, Box
import numpy as np

env = gym.make("Pendulum-v0")
env = gym.make("CartPole-v1")

num_actions = 0
num_inputs = env.observation_space.shape

box_min = None
box_max = None
a_taken_shape = None
model = None

if isinstance(env.action_space, Discrete):
    num_actions = env.action_space.n
    model = "softmax"

if isinstance(env.action_space, Box):
    num_actions = env.action_space.shape[0]
    a_taken_shape = None
    box_min = env.action_space.low
    box_max = env.action_space.high
    model = "gauss"

lrA = 0.01
lrC = 0.1
gamma = 0.99


def build_softmax_act(X, name="softmax_act_net"):
    with tf.variable_scope(name):
        net = layers.fully_connected(inputs=X,
                                     num_outputs=8,
                                     activation_fn=tf.nn.relu,
                                     )
        out = layers.fully_connected(inputs=net,
                                     num_outputs=num_actions,
                                     activation_fn=tf.nn.softmax,
                                     )
        # todo: juknout na to jak to facha a zda to fachas :*
        action = tf.multinomial(tf.log(out), 1)
        action = tf.cast(action, tf.int32)
        action = tf.squeeze(action)
        return action, out


def build_gauss_act(X, name="gauss_act_net",):
    with tf.variable_scope(name):
        net = layers.fully_connected(inputs=X,
                                     num_outputs=8,
                                     activation_fn=tf.nn.relu,
                                     )
        mu = layers.fully_connected(inputs=net,
                                    num_outputs=num_actions,
                                    activation_fn=None,
                                    )
        sigma = layers.fully_connected(inputs=net,
                                       num_outputs=num_actions,
                                       activation_fn=None,
                                       )

        sigma = tf.nn.softplus(sigma) + 1e-5

        gauss = tf.distributions.Normal(mu, sigma)

        action = tf.squeeze(gauss.sample(1))
        action = tf.clip_by_value(action, box_min, box_max)

        return action, gauss


def build_state_value_graph(X, name="state_v_net"):
    with tf.variable_scope(name):
        net = layers.fully_connected(inputs=X,
                                     num_outputs=8,
                                     activation_fn=tf.nn.relu,
                                     )
        net = layers.fully_connected(inputs=net,
                                     num_outputs=1,
                                     activation_fn=None
                                     )
        net = tf.squeeze(net)
        return net


def build_act(inp_ph, a_taken_ph, ret_ph, state_value, model_type):
    if model_type == "softmax":
        action, out = build_softmax_act(inp_ph)
        entropy = 0
        a_taken = tf.cast(a_taken_ph, tf.int32)
        action_one_hot = tf.one_hot(a_taken,
                                    depth=num_actions,
                                    axis=-1,
                                    dtype=tf.float32,
                                    )
        prob_out = tf.reduce_max(action_one_hot * out, 1)

    elif model_type == "gauss":
        action, out = build_gauss_act(inp_ph)
        prob_out = out.prob(a_taken_ph)
        entropy = tf.squeeze(out.entropy())
        prob_out = tf.squeeze(prob_out)

    else:
        print("Model type name error")
        return
    target = ret_ph - tf.stop_gradient(state_value)
    loss = -tf.log(prob_out + 1e-12) * target - 1e-1 * entropy
    loss = tf.reduce_mean(loss, 0)
    optimizer = tf.train.AdamOptimizer(learning_rate=lrA)
    minimize = optimizer.minimize(loss)

    return action, loss, minimize


def build_crit(inp_ph, ret_ph):
    state_value = build_state_value_graph(inp_ph)
    error = state_value - ret_ph
    loss = 0.5 * tf.square(error)
    loss = tf.reduce_mean(tf.square(loss), 0)

    optimizer = tf.train.AdamOptimizer(learning_rate=lrC)
    minimize = optimizer.minimize(loss)

    return state_value, loss, minimize


def build_graph(model_type="softmax"):
    inp_ph = tf.placeholder(dtype=tf.float32,
                            shape=(None, num_inputs[0]),
                            name="inp_ph",
                            )

    ret_ph = tf.placeholder(dtype=tf.float32,
                            shape=None,
                            name="total_return",
                            )
    a_taken_ph = tf.placeholder(dtype=tf.float32,
                                shape=None,
                                name="a_taken_ph",
                                )

    state_value, loss_c, minimize_c = build_crit(inp_ph, ret_ph)
    action, loss_a, minimize_a = build_act(inp_ph, a_taken_ph, ret_ph, state_value, model_type)

    loss = (loss_c, loss_a)
    minimize = (minimize_a, minimize_c)

    return inp_ph, ret_ph, a_taken_ph, action, minimize, loss


def main_loop():
    with tf.Session() as sess:
        inp_ph, ret_ph, a_taken_ph, action, minimize, loss = build_graph(model)
        sess.run(tf.global_variables_initializer())
        steps = 0
        ep = 0
        while steps < 1000000:
            s = env.reset()
            R = 0
            done = False
            all_states, rewards, rs_disc, a_taken = [], [], [], []
            ep += 1
            while not done:
                if ep % 100 == 0:
                    env.render()

                a = sess.run(action, feed_dict={inp_ph: [s]})

                s_, r, done, _ = env.step(a)

                all_states.append(s)
                a_taken.append(a)
                rewards.append(r)

                s = s_
                steps += 1

            for rew in rewards[::-1]:
                R = rew + gamma*R
                rs_disc.append(R)

            rs_disc = np.array(rs_disc[::-1])

            curr_loss = sess.run([loss, minimize],
                                 feed_dict={inp_ph: all_states, ret_ph: rs_disc, a_taken_ph: a_taken})[:1]

            if ep % 20 == 0:
                print("cur_loss:", curr_loss)
                print("episode: {1:3d}  curr_reward: {0:3f}".format(sum(rewards),
                                                                    ep))


if __name__ == "__main__":
    main_loop()
