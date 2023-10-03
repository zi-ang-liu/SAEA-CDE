import numpy as np
import matplotlib.pyplot as plt
from evaluation import evaluate
from SurrogateModel import build_model

class CDE:
    def __init__(self, policy, n_facility, n_var, n_fe, popsize=20, ub=1, lb=-1, F=0.8, CR=0.5, plot_graph=False) -> None:
        self.policy = policy 
        self.n_facility = n_facility
        self.n_var = n_var
        self.n_fe = n_fe
        self.popsize = popsize

        self.n_strategy = 3
        self.pbest_rate = 0.1
        self.F = F
        self.CR = CR
        self.plot_graph = plot_graph
        
        if not isinstance(ub, list):
            self.ub = np.array([ub] * self.n_var)
            self.lb = np.array([lb] * self.n_var)
    
    def run(self):
        
        # initialization
        X = self.lb + (self.ub - self.lb) * \
            np.random.rand(self.popsize, self.n_var)

        V = np.zeros((self.n_strategy, self.popsize, self.n_var))
        U = np.zeros((self.n_strategy, self.popsize, self.n_var))
        val_U_model = np.zeros((self.n_strategy, self.popsize))
        U_final = np.zeros((self.popsize, self.n_var))
        val_U_final_model = np.zeros((self.popsize, ))
        
        n_facility = self.n_facility
        for i in range(self.popsize):
            for j in range(n_facility):
                if X[i,j]>X[i,j+n_facility]:
                    X[i,j],X[i,j+n_facility]=X[i,j+n_facility],X[i,j]

        val_X = np.zeros((self.popsize, 1))
        val_U = np.ones((self.popsize, 1)) * np.inf

        for i in range(self.popsize):
            val_X[i] = evaluate(X[i, :], self.policy)

        fe = self.popsize

        self.x_train = X.copy()
        self.y_train = val_X.flatten().copy()
        model = build_model(self.x_train, self.y_train)

        con_graph = np.zeros((self.n_fe))
        con_graph[0:fe] = min(val_X)

        # accuary
        accuracy_rate = np.zeros((self.n_fe))
        n_success = 0
        n_total = 0

        while fe <= self.n_fe-1:
            
            # mutation
            idx = np.arange(0,self.popsize).reshape((self.popsize,1))
            idx_r1r2r3 = np.random.randint(0, self.popsize, size=(self.popsize, 3))
            for i in range(self.popsize): # vectorization?
                while ([i]==idx_r1r2r3[i,:]).any():
                    idx_r1r2r3[i,:] = np.random.randint(0, self.popsize, size=(1, 3))

            random_idx = np.hstack((idx,idx_r1r2r3))
            idx, r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2], random_idx[:, 3]

            # V[i]=X[r1]+F(X[r2]-X[r3])
            V[0,:,:] = X[r1, :] + self.F * (X[r2, :] - X[r3, :])

            # current-to-pbest/2: 
            pbest_num = np.max((np.round(self.pbest_rate*self.popsize),2)).astype(int)
            pbest_index = np.random.choice(pbest_num)
            sorted_index = np.argsort(self.y_train)
            pbest = self.x_train[sorted_index[pbest_index], :]
            V[1,:,:] = X[idx, :] + self.F * (pbest - X[idx, :]) + self.F * (X[r1, :] - X[r2, :])

            # best/1:
            V[2,:,:] = self.x_train[sorted_index[0], :] + self.F * (X[r1, :] - X[r2, :])

            mask_bound = np.random.uniform(low=self.lb, high=self.ub, size=(self.n_strategy, self.popsize, self.n_var))
            V = np.where(V < self.lb, mask_bound, V)
            V = np.where(V > self.ub, mask_bound, V)

            # crossover
            mask_cr = np.random.rand(self.n_strategy, self.popsize, self.n_var) < self.CR
            jrand =np.random.randint(0, self.n_var, size=(self.n_strategy, self.popsize))
            for k in range(self.n_strategy):
                for i in range(self.popsize): # vectorization?
                    mask_cr[k, i,jrand[k,i]] = True
            U = np.where(mask_cr, V, X)

            # repair
            n_facility = self.n_facility
            for k in range(self.n_strategy):
                for i in range(self.popsize):
                    for j in range(n_facility):
                        if U[k,i,j]>U[k,i,j+n_facility]:
                            U[k,i,j],U[k,i,j+n_facility]=U[k,i,j+n_facility],U[k,i,j]
            
            # surrogate evaluation
            for k in range(self.n_strategy):
                val_U_model[k,:] = model.predict(U[k,:,:])
            U_idx = np.argmin(val_U_model,axis=0)
            for i in range(self.popsize):
                U_final[i,:] = U[U_idx[i], i, :]
                val_U_final_model[i] = val_U_model[U_idx[i], i]
            pop_idx = np.asarray(np.where(val_U_final_model<val_X.flatten())).flatten()
            
            if pop_idx.size != 0:
                # real function
                for i in range(pop_idx.size):
                    val_U[pop_idx[i]] = evaluate(U_final[pop_idx[i], :], self.policy)

                # re-build surrogate
                self.x_train = np.vstack((self.x_train, U_final[pop_idx,:])) 
                self.y_train = np.hstack((self.y_train, val_U[pop_idx].flatten()))
                model = build_model(self.x_train, self.y_train)

                # accuary
                n_total = n_total + pop_idx.size
                n_success = n_success + (val_U<val_X).sum()
                accuracy_rate[fe:fe+pop_idx.size] = n_success/n_total
                # print(n_success/n_total)
            
                # selection
                X = np.where(val_U<val_X, U_final, X)
                val_X = np.where(val_U<val_X, val_U, val_X)

                index = np.argmin(val_X)
                best = X[index,:]
                best_val = val_X[index]

                con_graph[fe:fe+pop_idx.size] = best_val
                fe += pop_idx.size
                print('number of evaluation: ' + str(fe), end='\r')

        if self.plot_graph == True:
            plt.plot(con_graph)
            plt.show()

        print('Computation completed successfully.')

        return best, best_val, con_graph, accuracy_rate

if __name__ == '__main__':
    algo = CDE(n_facility=6, popsize=20, n_iter=30, plot_graph=True, policy='ss_policy')
    solution, function_value = algo.run()
    print(solution)
    print(function_value)
