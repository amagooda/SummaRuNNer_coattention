import SummRunner_V2.HelpingFunctions as hF
from lxml import etree as ET
from tqdm import tqdm
import codecs
import os

def load_data(data_file_path):
    translate_file = codecs.open('./checkpoint/forum_to_translate.txt', 'w', encoding='utf8')

    tree = ET.parse(data_file_path)
    root = tree.getroot()
    for post_item in tqdm(root):
        post_id = post_item.tag
        translate_file.write('@START_Art@ @{}@\n'.format(post_id))

        title = [item.text for item in post_item.findall('Title')][0]
        words = hF.tokenize_text(title)
        translate_file.write(' '.join(words) + '\n')

        initial_comment = [item.text for item in post_item.findall('Body')][0].split('\n')
        for sentence in initial_comment:
            words = hF.tokenize_text(sentence)
            translate_file.write(' '.join(words) + '\n')


        translate_file.write('@COMMENTS@\n')
        for comment_item in post_item.findall('Comment'):
            comment_body = [item.text for item in comment_item.findall('Body')][0].split('\n')
            for sentence in comment_body:
                words = hF.tokenize_text(sentence)
                translate_file.write(' '.join(words) + '\n')

        translate_file.write('@HIGHLIGHT@\n')
        selected_sentences = [item.text for item in post_item.findall('selected_sentences')][0].split('\n')
        for sentence in selected_sentences:
            words = hF.tokenize_text(sentence)
            translate_file.write(' '.join(words) + '\n')

        translate_file.flush()
    translate_file.close()


def load_cnn_dm_data(data_parent_dir):
    import json
    translate_file = codecs.open('./checkpoint/cnn_to_translate.txt', 'w', encoding='utf8')
    for data_dir in ['test', 'train', 'val']:
        file_names = os.listdir(data_parent_dir + '//' + data_dir)
        for fname in tqdm(file_names):
            with open(data_parent_dir + '//' + data_dir + '//' + fname) as json_file:
                data = json.load(json_file)
                article = [x.strip() for x in data['article']]
                if len(article) < 2:
                    continue
                summary = [article[x] for x in data['extracted']]

                translate_file.write('@START_Art@ @{}@\n'.format(fname))
                translate_file.write(article[0] + '\n')
                translate_file.write('@COMMENTS@\n')
                for line in article:
                    translate_file.write(line + '\n')
                translate_file.write('@HIGHLIGHT@\n')
                for line in summary:
                    translate_file.write(line + '\n')
            translate_file.flush()
    translate_file.close()



if __name__ == '__main__':
    # load_cnn_dm_data('E:/Work/Summarization_Codes/Fast_abs_rl/data/cnn_dailymail/finished_files/')
    load_data('E:/Work/Summarization_samples/SummRunner_V2/forum_data/data_V2/Parsed_Data.xml')
