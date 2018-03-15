import click
import readn
import time


@click.command()
@click.option('--n_stop', default=100)
@click.option('--cfile', default='/home/cbe-master/realfast/soft/vysmaw_apps/vys.conf')
def vyscheck(n_stop, cfile):
    types = readn.run(n_stop, cfile=cfile)
    for key in list(types.keys()):
        print('Received {0} vys messages of type {1}'.format(types[key], key))

    time.sleep(1)
