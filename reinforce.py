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

lr = 0.01
gamma = 0.99


def build_softmax_act(X, name="softmax_act_net"):
    with tf.variable_scope(name):
        net = layers.fully_connected(inputs=X,
                                     num_outputs=128,
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
        mu = layers.fully_connected(inputs=X,
                                    num_outputs=num_actions,
                                    activation_fn=None,
                                    )
        sigma = layers.fully_connected(inputs=X,
                                       num_outputs=num_actions,
                                       activation_fn=None,
                                       )

        sigma = tf.nn.softplus(sigma) + 1e-5

        gauss = tf.distributions.Normal(mu, sigma)

        action = tf.squeeze(gauss.sample(1))
        action = tf.clip_by_value(action, box_min, box_max)

        return action, gauss


def build_graph(model_type="softmax"):
    inp_ph = tf.placeholder(dtype=tf.float32,
                            shape=(None, num_inputs[0]),
                            name="inp_ph",
                            )

    if model_type == "softmax":
        action, out = build_softmax_act(inp_ph)

    if model_type == "gauss":
        action, out = build_gauss_act(inp_ph)

    ret_ph = tf.placeholder(dtype=tf.float32,
                            shape=None,
                            name="total_return",
                            )
    a_taken_ph = tf.placeholder(dtype=tf.float32,
                                shape=None,
                                name="a_taken_ph",
                                )
    if model_type == "softmax":
        entropy = 0
        a_taken = tf.cast(a_taken_ph, tf.int32)
        action_one_hot = tf.one_hot(a_taken,
                                    depth=num_actions,
                                    axis=-1,
                                    dtype=tf.float32,
                                    )
        prob_out = tf.reduce_max(action_one_hot * out, 1)
    elif model_type == "gauss":
        prob_out = out.prob(a_taken_ph)
        entropy = tf.squeeze(out.entropy())
        prob_out = tf.squeeze(prob_out)

    else:
        print("Model type name error")
        a_taken_ph = None
        prob_out = None

    loss = -tf.log(prob_out+1e-12) * ret_ph - 1e-1*entropy
    loss = tf.reduce_mean(loss, 0)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    minimize = optimizer.minimize(loss)

    return inp_ph, ret_ph, a_taken_ph, action, minimize, loss, out, prob_out


def main_loop():
    with tf.Session() as sess:
        inp_ph, ret_ph, a_taken_ph, action, minimize, loss, out, pr_out = build_graph(model)
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

            rs_disc = rs_disc[::-1]
            rs_disc = np.array(rs_disc)
            rs_disc = (rewards-np.mean(rs_disc))/np.std(rs_disc)

            curr_loss = sess.run([loss, pr_out, minimize],
                                 feed_dict={inp_ph: all_states, ret_ph: rs_disc, a_taken_ph: a_taken})[:1]

            if ep % 20 == 0:
                print(curr_loss)
                print("episode: {2:3d}   curr_los: {0:.3f}   curr_reward: {1:3f}".format(curr_loss[0],
                                                                                         sum(rewards),
                                                                                         ep))
            # if ep % 20 == 0:
            #     print("episode: {1:3d}   curr_reward: {0:3f}".format(sum(rewards), ep))


if __name__ == "__main__":
    main_loop()
