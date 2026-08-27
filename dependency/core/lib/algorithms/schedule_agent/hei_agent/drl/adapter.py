class Adapter:
    @staticmethod
    def done_adapter(done, t):
        if t % 100 == 0:
            return not done
        else:
            return done
