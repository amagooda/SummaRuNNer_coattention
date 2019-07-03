from pyrouge import Rouge155
from pyrouge.utils import log
import re
import os
from os.path import join
import logging
import tempfile
import subprocess as sp
from cytoolz import curry
import codecs
import shutil

_ROUGE_PATH = '/mnt/e/Work/Summarization_Codes/pyrouge/tools/ROUGE-1.5.5'


def read_summaries(file_path):
    lines = []
    data_reader = open(file_path, 'r')
    for line in data_reader:
        line = line.replace('\n', '').replace('\r', '').strip()
        lines.append(line)
    return lines


def write_in_files(output_dir, data, extension):
    for index, summary in enumerate(data):
        if len(summary) == 1:
            data_writer = codecs.open(output_dir + '/{}.{}'.format(index + 1, extension), 'w', encoding='utf8')
            data_writer.write(summary[0])
            data_writer.close()
        elif len(summary) == 2:
            data_writer = codecs.open(output_dir + '/{}_1.{}'.format(index + 1, extension), 'w', encoding='utf8')
            data_writer.write(summary[0])
            data_writer.close()

            data_writer = codecs.open(output_dir + '/{}_2.{}'.format(index + 1, extension), 'w', encoding='utf8')
            data_writer.write(summary[1])
            data_writer.close()


def eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir, dir_name,
               cmd='-c 95 -r 1000 -n 2 -m', system_id=1):
    """ evaluate by original Perl implementation"""
    # silence pyrouge logging
    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    # with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_dir = '/mnt/e/Work/Summarization_samples/SummRunner_V2/output/{}/temp/'.format(dir_name)

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    Rouge155.convert_summaries_to_rouge_format(
        dec_dir, join(tmp_dir, 'dec'))
    Rouge155.convert_summaries_to_rouge_format(
        ref_dir, join(tmp_dir, 'ref'))
    Rouge155.write_config_static(
        join(tmp_dir, 'dec'), dec_pattern,
        join(tmp_dir, 'ref'), ref_pattern,
        join(tmp_dir, 'settings.xml'), system_id
    )
    cmd = ('sudo perl ' + _ROUGE_PATH + '/ROUGE-1.5.5.pl'
           + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
           + cmd
           + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
    output = sp.check_output(cmd, universal_newlines=True, shell=True)
    return output


def main():
    max_rouge_index = 0
    max_rouge = 0
    max_output = ''
    dir_name = '20_25_forum_tune_guf'
    print('Evaluatiig {} ....'.format(dir_name))
    for i in range(50):
        dec_dir = '/mnt/e/Work/Summarization_samples/SummRunner_V2/output/{}/test_{}/dec/'.format(dir_name, i)
        ref_dir = '/mnt/e/Work/Summarization_samples/SummRunner_V2/output/{}/test_{}/ref_abs/'.format(dir_name, i)

        if not os.path.exists('/mnt/e/Work/Summarization_samples/SummRunner_V2/output/{}/test_{}/dec/'.format(dir_name, i)):
            continue

        dec_pattern = r'(\d+).dec'
        ref_pattern = '#ID#.ref'
        print('test_{}'.format(i))
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir, dir_name)
        print(output)

        lines = output.split('\n')

        current_rouge = 0
        for line in lines:
            if 'Average_F' in line:
                val = float(line.split('Average_F:')[1].split('(')[0].strip())
                current_rouge += val
        if current_rouge > max_rouge:
            max_output = output
            max_rouge = current_rouge
            max_rouge_index = i
    print('Best Model...... step {}'.format(max_rouge_index))
    print(max_output)


main()