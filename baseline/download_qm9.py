import urllib.request

url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv'
output_file = 'qm9.csv'

urllib.request.urlretrieve(url, output_file)
print('File downloaded successfully.')