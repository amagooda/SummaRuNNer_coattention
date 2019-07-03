import torch
from torch.utils import data
import os
import pickle

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_set_path, use_back_translation=False):
        if os.path.exists(data_set_path):
            with open(data_set_path, "rb") as output_file:
                if use_back_translation is True:
                    [posts, comments, answers, human_summaries, sentence_strs, posts_translated, comments_translated] = pickle.load(output_file)
                    self.posts_translated = posts_translated
                    self.comments_translated = comments_translated
                else:
                    [posts, comments, answers, human_summaries, sentence_strs] = pickle.load(output_file)
                    self.posts_translated = None
                    self.comments_translated = None
            
            self.use_back_translation = use_back_translation
            self.posts = posts
            self.comments = comments
            self.human_summaries = human_summaries
            self.answers = answers
            self.sentence_strs = sentence_strs
        else:
            print('{} doesn\'t exist please make sure path is correct')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.posts)

    def __getitem__(self, index):
        post = self.posts[index]
        comment = self.comments[index]
        human_summary = self.posts[index]
        answer = self.answers[index]
        sentence_str = self.sentence_strs[index]
        if self.use_back_translation is True:
            post_translated = self.posts_translated[index]
            comment_translated = self.comments_translated[index]
            return post, comment, human_summary, answer, sentence_str, post_translated, comment_translated
        else:
            post_translated= None
            comment_translated = None
            return post, comment, human_summary, answer, sentence_str, post, comment
        
#         return post, comment, human_summary, answer, sentence_str, post_translated, comment_translated