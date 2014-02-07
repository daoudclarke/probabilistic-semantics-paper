from learn import Learner
from learn_test import train



def run():
    train_data = train()
    data = [train_data[0], train_data[5]]
    learner = Learner()
    learner.max_its = 3
    learner.learn(data)


if __name__ == "__main__":
    import cProfile
    import pstats

    cProfile.run('run()', 'profile_stats')
    p = pstats.Stats('profile_stats')
    p.sort_stats('cumulative').print_stats(50)
