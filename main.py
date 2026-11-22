from sklearn.model_selection import train_test_split
from model import *
from function import *
from utils import *

if __name__ == '__main__':

    data = pd.read_csv('data/data.csv')
    label = pd.read_csv('data/label.csv')

    for valr in [0.8]:  # train dataset ratio
        print('Train ratio:%.2f' % valr)
        X_train, X_test, Y_train, Y_test = train_test_split(data.values, label.flag.values,
                                                            test_size=1 - valr,
                                                            random_state=2017,
                                                            stratify=label.flag.values
                                                            )

        data_len = X_train.shape[1]
        for i in range(20):
            print('Round: %d' % i)
            train_gen = BalancedBatchGenerator(
                X_train,
                Y_train,
                batch_size=64,
                pos_fraction=float(np.mean(Y_train == 1)),  # set the positive-to-negative sample ratio for each batch
                steps_per_epoch=int(np.ceil(len(Y_train) / 64)),
                seed=42
            )

            model = Hybrid_Model(data_len)  #Hybrid_Model

            if i == 0:
                print(model.summary())

            model.fit(
                train_gen,
                epochs=20,
                verbose=1,
                validation_data=(X_test, Y_test),
                callbacks=[ProMetric(valr, i, ([X_test], Y_test))]
            )
