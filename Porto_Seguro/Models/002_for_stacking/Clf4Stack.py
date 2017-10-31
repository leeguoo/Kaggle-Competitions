class Clf4Stack(object):
    def __init__(self, model, n_splits=5):
        self.n_splits = n_splits
        self.model = model

    def fit_predict(self, trainX, trainy, testX):

        self.train4stack = np.zeros(len(trainX))
        self.test4stack = np.zeros(len(testX))

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=44)

        for train_index, test_index in skf.split(trainX, trainy):
            X_train, X_test = trainX[train_index], trainX[test_index]
            y_train, y_test = trainy[train_index], trainy[test_index]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict_proba(X_test)[:,1]
            self.train4stack[test_index] = y_pred
            self.test4stack += self.model.predict_proba(testX)[:,1]
        
        self.test4stack /= self.n_splits
            
    def output(self,train_file_name='train4stack.csv',
                    test_file_name='test4stack.csv',
                    col_name='F4stack'):

        pd.DataFrame({col_name:self.train4stack}).to_csv(train_file_name,index=False) 
        pd.DataFrame({col_name:self.test4stack}).to_csv(test_file_name,index=False)
