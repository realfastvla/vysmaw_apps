import click
import readn
import time

# new style
message_types = ["VYSMAW_MESSAGE_SPECTRA", "VYSMAW_MESSAGE_QUEUE_ALERT",
                 "VYSMAW_MESSAGE_SPECTRUM_BUFFER_STARVATION",
                 "VYSMAW_MESSAGE_SIGNAL_RECEIVE_FAILURE",
                 "VYSMAW_MESSAGE_VERSION_MISMATCH",
                 "VYSMAW_MESSAGE_SIGNAL_RECEIVE_QUEUE_UNDERFLOW",
                 "VYSMAW_MESSAGE_END"]


@click.command()
@click.option('--n_stop', default=100)
@click.option('--cfile', default='/home/cbe-master/realfast/soft/vysmaw_apps/vys.conf')
def vyscheck(n_stop, cfile):
    types = readn.run(n_stop, cfile=cfile)
    for i in range(len(types)):
        print('Received {0} vys messages of type {1}'.format(types[i], message_types[i]))

    time.sleep(1)
