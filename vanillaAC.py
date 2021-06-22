import tensorflow as tf
import tensorflow.contrib.layers as layers
import gym
from gym.spaces import Discrete, Box
import numpy as np

env = gym.make("Pendulum-v0")
env = gym.make("CartPole-v1")
env = gym.make("MountainCarContinuous-v0")

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
td_n = 64


def linear_map(val, A, B, a, b):
    val = np.array(val)
    slope = (a-b)/(A-B)
    return (val - A)*slope + a


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


def build_act(inp_ph, a_taken_ph, td_error, model_type, lr):
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

    loss = -tf.log(prob_out + 1e-12) * tf.stop_gradient(td_error) - 1e-1 * entropy
    loss = tf.reduce_mean(loss, 0)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    minimize = optimizer.minimize(loss)

    return action, loss, minimize


def build_crit(inp_ph, target_ph, lr):
    state_value = build_state_value_graph(inp_ph)
    td_error = target_ph - state_value
    loss = 0.5 * tf.square(td_error)
    loss = tf.reduce_mean(loss, 0)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    minimize = optimizer.minimize(loss)

    return state_value, loss, minimize, td_error


def build_graph(model_type="softmax", lr_a=0.01, lr_c=0.1):
    inp_ph = tf.placeholder(dtype=tf.float32,
                            shape=(None, num_inputs[0]),
                            name="inp_ph",
                            )

    target_ph = tf.placeholder(dtype=tf.float32,
                               shape=None,
                               name="total_return",
                               )
    a_taken_ph = tf.placeholder(dtype=tf.float32,
                                shape=None,
                                name="a_taken_ph",
                                )

    state_value, loss_c, minimize_c, td_error = build_crit(inp_ph, target_ph, lr_c)
    action, loss_a, minimize_a = build_act(inp_ph, a_taken_ph, td_error, model_type, lr_a)

    loss = (loss_c, loss_a)
    minimize = (minimize_a, minimize_c)

    return inp_ph, target_ph, a_taken_ph, action, state_value, minimize, loss


def solve(lr_a, lr_c, verbosity=1):
    tf.reset_default_graph()
    inp_ph, target_ph, a_taken_ph, action, s_val, minimize, loss = build_graph(model, lr_a, lr_c)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        result = 0
        steps_total = 100000
        steps = 0
        ep = 0

        while steps < steps_total:
            s = env.reset()
            done = False
            ep += 1
            rewards, states, actions, disc_rs = [], [], [], []
            R = 0
            while not done:
                if ep % 100 == 0:
                    env.render()

                a, val = sess.run([action, s_val], feed_dict={inp_ph: [s]})

                s_, r, done, _ = env.step(a)

                states.append(s)
                actions.append(a)
                rewards.append(r)

                s = s_
                steps += 1
                R += r

                if len(rewards) == td_n+1 and not done:
                    s_train, a_train = states.pop(0), actions.pop(0)
                    disc_r = 0
                    for rew in rewards[::-1]:
                        disc_r = rew + gamma * disc_r
                    rewards.pop(0)

                    next_val = sess.run(s_val, feed_dict={inp_ph: [s]})

                    target = disc_r + gamma ** (td_n+1) * next_val
                    feed_dict = {inp_ph: [s_train],
                                 target_ph: [target],
                                 a_taken_ph: [a_train]
                                 }

                    sess.run(minimize, feed_dict)

            disc_r = 0
            for rew in rewards[::-1]:
                disc_r = rew + gamma*disc_r
                disc_rs.append(disc_r)

            target = np.array(disc_rs[::-1])
            feed_dict = {inp_ph: np.array(states),
                         target_ph: target,
                         a_taken_ph: np.array(actions),
                         }

            curr_loss = sess.run([loss, minimize], feed_dict)[:1]

            if ep % 20 == 0 and verbosity > 0:
                print(curr_loss)
                print("episode: {1:3d} curr_reward: {0:3f}".format(R,
                                                                   ep))

            if steps > steps_total-10000:
                result += R

        return result/10000


def random_search(samples_num):
    results = []
    for _ in range(samples_num):
        params = np.random.random(2)
        params = linear_map(params, 0, 1, 0.0005, 0.3)

        result = solve(params[0], params[1])
        results.append((params, result))
    return results


def main_loop():

    results = random_search(20)
    # solve(lrA, lrC)
    return results


if __name__ == "__main__":
    res = main_loop()
