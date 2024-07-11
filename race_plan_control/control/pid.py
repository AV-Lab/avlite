import logging 
class Controller:

    def __init__(self, alpha=0.1, beta=0.01, gamma=0.01): 
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.past_cte = []


    def control(self, cte):
        self.past_cte.append(cte)

        if len(self.past_cte) < 2:
            return -self.alpha * cte
        return -self.alpha * cte + self.beta * sum(self.past_cte) + self.gamma * (cte - self.past_cte[-2])


if __name__ == "__main__":
    print("Hello World")