class EvomanEA(object):
    """
    EvomanEA is a common interface for any Evolutionary computing solution for Evoman
    Implementations should implement this interface.

     1) For training:

            env = Environment(
                    playermode="ai",
                    enemymode="static",
                )

            ea = EvomanSimple()
            ea.train(env)
            ea.save(filename)

     2) For running:

            env = Environment(
                    playermode="ai",
                    enemymode="static",
                )

            ea = EvomanSimple()
            ea.load(filename)
            ea.run(env)
    """

    def train(self, env):
        """
        Method that uses the environment to run experiments in order to evolve the controller
        Args:
            env: Environment to use
        Returns:

        """
        pass

    def run(self, env):
        """
        Method that calls env.play() passing proper controller context
        Args:
            env: Environment to use

        Returns:

        """
        pass

    def save(self, filename):
        """
        Method that saves the best controller to the file
        Args:
            filename: name to use in order to save the file

        Returns:

        """
        pass

    def load(self, filename):
        """
        Method that loads the controller from the file
        Args:
            filename: name of the controller's file

        Returns:

        """
        pass
