"""
NAIST誤用コーパス中の助詞の誤りのみを抽出する
出力中にまだ<goyo>タグがのこるので、手で直す
"""
import re
import argparse
import MySQLdb

# regex
# target_tags = '(om|ga|wo|ni|de)/(ga|wo|ni|de)'  # がをにで不足
target_tags = '(ga|wo|ni|de)/(ga|wo|ni|de)'  # がをにで
target_tags_regex = r"<goyo crr='(.)' type='p\/{}'>(.*?)<\/goyo>".format(target_tags)
tags_regex = r"<goyo crr='(.*?)'( .*?)>.*?<\/goyo>"
some_tags_regex = r"<goyo crr1='(.*?)' crr2='(.*?)' (type|type1)='(?!p\/.*)(.*?)'( .*?)?>.*?<\/goyo>"

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
        SELECT annotated_sentence, correct_sentence FROM data 
        WHERE annotated_sentence RLIKE "type='p/{}'"
        AND correction_id=1
    """.format(target_tags)
    c.execute(sql)
    return c.fetchall()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save-crr', required=True, help='Save file for correct sentences')
    parser.add_argument('--save-err', required=True, help='Save file for error sentences')
    args = parser.parse_args()
    f_crr = open(args.save_crr, 'w')
    f_err = open(args.save_err, 'w')

    result = select_particle_errors()
    for r in result:
        annotated_sent = r[0]
        correct_sent = r[1]
        # 助詞だけ間違ったままにしてタグを削除
        target_sub_sent = re.sub(target_tags_regex, r'\4', annotated_sent)
        # 他の誤りを正しく置換してタグを削除
        tags_sub_sent = re.sub(tags_regex, r'\1', target_sub_sent)
        tags_sub_sent = re.sub(some_tags_regex, r'\1', tags_sub_sent)
        error_sent = tags_sub_sent

        if len(error_sent) != len(correct_sent):
            continue

        print(correct_sent, file=f_crr)
        print(error_sent, file=f_err)

    f_crr.close()
    f_err.close()

if __name__ == '__main__':
    main()

