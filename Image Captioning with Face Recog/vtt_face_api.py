import requests


def vtt_face_recognize(image_path, url=None, data=None):
    name_list = list()
    conf_list = list()
    roi_list = list()

    # Fill up and Modify your face recognition APIs
    # input
    #   image path
    # output
    #   name list : class name list for face recognition
    #   confidence list : confidence result for face recognition
    #   roi list : detected face boxes (left-top x, left-top y, right-bottom x, right-bottom y)
    if url is None:
        url = ''
    if data is None:
        data = {"modules": "friends.face", }
    files = {'image': (image_path, open(image_path, 'rb'), "multipart/form-data")}
    response = requests.post(url, files=files, data=data)

    if response.status_code == 201:
        response_data = response.json()
        res = response_data['results'][0]['module_result']
        num_box = len(res)
        for i in range(num_box):
            name_list.append(res[i]['label'][0]['description'])
            conf_list.append([res[i]['label'][0]['score']])
            x1 = int(res[i]['position']['x'])
            y1 = int(res[i]['position']['y'])
            x2 = x1 + int(res[i]['position']['w'])
            y2 = y1 + int(res[i]['position']['h'])
            roi_list.append([x1, y1, x2, y2])

    return name_list, conf_list, roi_list
