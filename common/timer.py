import time


class Timer:
    """
    Simple wrapper for time.process_time().
    """

    def __init__(self):
        """
        Initialize and start timer.
        """

        self.start = time.process_time()
        """ (float) Seconds. """

    def reset(self):
        """
        Reset timer.
        """

        self.start = time.process_time()

    def elapsed(self):
        """
        Get elapsed time in seconds

        :return: elapsed time in seconds
        :rtype: float
        """

        return (time.process_time() - self.start)


def elapsed(function):
    """
    Time a function call.

    :param function: function to call
    :type function: callable
    """

    assert callable(function)

    timer = Timer()
    results = function()
    time = timer.elapsed()

    if results is None:
        results = time
    elif isinstance(results, tuple):
        results = tuple(list(results) + [time])
    else:
        results = (results, time)

    return results
