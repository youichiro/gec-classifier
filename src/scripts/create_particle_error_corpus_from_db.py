# -*- coding: utf-8 -*-
"""
NAIST誤用コーパス中の助詞の誤りのみを抽出する
出力中にまだ<goyo>タグがのこるので、手で直す
"""
import re
import argparse
import MySQLdb

# regex
# target_tags = '(ga|wo|ni|de)/(ga|wo|ni|de)'  # がをにで
target_tag = '(ga|no|wo|ni|he|to|yori|kara|de|ya|wa|niwa|karawa|towa|dewa|hewa|madewa|yoriwa|made|om)/(ga|no|wo|ni|he|to|yori|kara|de|ya|wa|niwa|karawa|towa|dewa|hewa|madewa|yoriwa|made|ad)'  # 不足，余剰も含める
target_tag_regex = r"<goyo crr='([^(<\/goyo>)]*?)' type='p\/{}'>(.*?)<\/goyo>".format(target_tag)
target_multi_tag_regex = r"<goyo crr1='(.*?)' crr2='(.*?)' type1='p\/{}' type2='(.*?)'>(.*?)<\/goyo>".format(target_tag)
goyo_tag_regex = r"<goyo crr='([^(<\/goyo>)]*?)'( .*?)>.*?<\/goyo>"
some_tags_regex = r"<goyo crr1='(.*?)' crr2='(.*?)' (type|type1)='(?!p\/.*)(.*?)'( .*?)?>.*?<\/goyo>"
valid_tag_regex = r"<goyo crr='' type='p\/om\/(.*?)'><\/goyo>"

# mysql
conn = MySQLdb.connect(
    db='naist_goyo_corpus',
    user='gakkai',
    passwd='gakkai',
    charset='utf8mb4'
)
c = conn.cursor()


def select_particle_errors():
    sql = """
        SELECT annotated_sentence, correct_sentence
        FROM data
        WHERE annotated_sentence RLIKE "type='p/{}'"
        AND correction_id=1
    """.format(target_tag)
    c.execute(sql)
    return c.fetchall()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save-ans', required=True, help='Save file for correct sentences')
    parser.add_argument('--save-err', required=True, help='Save file for error sentences')
    args = parser.parse_args()
    f_ans = open(args.save_ans, 'w')
    f_err = open(args.save_err, 'w')
    continue_count = 0

    particle_error_sentences = select_particle_errors()
    for s in particle_error_sentences:
        annotated_sentence = s[0]
        correct_sentence = s[1]

        # 不適切なタグを削除
        annotated_sentence = re.sub(valid_tag_regex, r'', annotated_sentence)
        # 助詞だけ間違ったままにしてタグを削除
        error_sentence = re.sub(target_tag_regex, r'\4', annotated_sentence)
        error_sentence = re.sub(target_multi_tag_regex, r'\6', error_sentence)
        # 他の誤りを正しく置換してタグを削除
        error_sentence = re.sub(goyo_tag_regex, r'\1', error_sentence)
        error_sentence = re.sub(some_tags_regex, r'\1', error_sentence)

        if error_sentence == correct_sentence \
          or abs(len(error_sentence) - len(correct_sentence)) > 3:
            continue_count += 1
            print('Continued: ' + str(continue_count))
            print('[annotated_sentence]\n' + annotated_sentence)
            print('[error_sentence]\n' + error_sentence)
            print('[correct_sentence]\n' + correct_sentence)
            print()
            continue

        print(correct_sentence, file=f_ans)
        print(error_sentence, file=f_err)

    f_ans.close()
    f_err.close()

if __name__ == '__main__':
    main()

