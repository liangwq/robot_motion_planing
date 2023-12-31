#---------- definisco environment ----------------------------------
class Env:
    #costruttore
    def __init__(self):
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    #definisco boundary ostacoli (standard 2D)
    @staticmethod
    def obs_boundary():
        obs_boundary =  [0, 50, 0, 30]
        return obs_boundary

    #ostacoli rettangolari
    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [13, 11, 8, 8],
            #[18, 22, 8, 3],
            #[26, 7, 2, 12],
            [32, 14, 10, 2]
        ]
        return obs_rectangle

    #ostacoli circolari
    @staticmethod
    def obs_circle():
        obs_cir = [
            [7, 12, 3],
            #[46, 20, 2],
            #[15, 5, 2],
            [37, 7, 3],
            [37, 23, 3]
        ]

        return obs_cir
