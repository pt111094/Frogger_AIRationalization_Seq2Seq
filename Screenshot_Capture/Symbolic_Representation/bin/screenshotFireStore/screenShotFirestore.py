import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json
import xlsxwriter
from collections import Counter
from datetime import tzinfo, timedelta, datetime
import pytz
from PIL import Image
import os
# Use a service account

class dataset(object):
    def __init__(self, worker_id = None):
        self.worker_ID = None
        self.current_image = None
        self.next_image = None
        self.doc_id = None
def sort_key(data):
    print(int(data.worker_ID)*10000 + int(data.doc_id))
    return int(data.worker_ID)*10000 + int(data.doc_id)
# input is a list of rationalizations
def analyseRationalizations(rationalizations):
    oneString = ""
    rationalizationLength = []
    for sentence in rationalizations:
        new = sentence.replace("'", "")
        new = new.replace(".", " ")
        rationalizationLength.append(sentence.count(' ') + 1)
        oneString += new
    # split() returns list of all the words in the string
    split_it = oneString.split()

    # Pass the split_it list to instance of Counter class.
    print Counter(split_it)
    print "lengths of rationalizations:"
    print rationalizationLength
    print "average length of a rationalization " + str(sum(rationalizationLength)/len(rationalizationLength))

    # most_common() produces k frequently encountered
    # input values and their respective counts.

def strToDate(d,key):
    zz = d[key].split("datetime.datetime(")[1]
    zzz = zz.split(",")
    n = len(zzz)
    # print(n)
    zzz[n-1] = "UTC"
    for i in range(n-1):
        zzz[i] = int(zzz[i])
    # print(zzz)
    if n==8:
        return datetime(zzz[0],zzz[1],zzz[2],zzz[3],zzz[4],zzz[5],zzz[6],tzinfo=pytz.UTC)    
    elif n==7: 
        return datetime(zzz[0],zzz[1],zzz[2],zzz[3],zzz[4],zzz[5],0,tzinfo=pytz.UTC)



if __name__ == '__main__':
    cred = credentials.Certificate('Frogger-1866892a5544.json')
    firebase_admin.initialize_app(cred)

    db = firestore.client()

    users_ref = db.collection(u'sasr-turk')
    docs = users_ref.get()

    sessionID = db.collection(u'sessions-turk')
    # TODO: we only take these sasr
    sessions = sessionID.get()
    workIds = []
    # make 2 dictionaries to holder worker ID, secrete key
    # make 2 list to hold rationalizations and actions
    id = {}
    secreteKey = {}
    outputSecreteKey = []
    rationalizations = []
    actions = []
    nextState = []
    complete_data = {}
    start_times = []
    end_times = []
    finished_study = []
    game_start_times = []
    finished_game = []
    old_worker_ids = ["A12723JDRPT207","A1CH3TODZNQCES","A1D3QRG52OISSR","A1FOTRH3UJMKTS","A1GXFMAC759VRM","A1KEA2Z47S3UPI"
,"A1M682B2WUSYJP","A1OCEC1TBE3CWA","A1TGV7LT6LTIQU","A1X84T4EFW04GZ","A23782O23HSPLA","A2CNSIECB9UP05","A2F9V69F6TZIAB","A2IQ0QCTQ3KWLT","A2OFN0A5CPLH57","A2P1KI42CJVNIA","A2Q2A7AB6MMFLI"
,"A2WZ0RZMKQ2WGJ","A3EKETMVGU2PM9","A3GEL5PWFIK05S","A3GI86L18Z71XY","A3J55BJGV95JKH","A3J5UWZM4ZZ1D6","A3KSAP865D3L7D","A3SB7QYI84HYJT","A5EU1AQJNC7F2","AAAL4RENVAPML"
,"AIXTI8PKSX1D2"]
    for s in sessions:
        curr = str(s.to_dict())
        workIds.append(str(s.id))

        json_acceptable_string = curr.replace("u'", '"')

        json_acceptable_string = json_acceptable_string.replace("'", "\"")
        json_acceptable_string = json_acceptable_string.replace("False","false")
        json_acceptable_string = json_acceptable_string.replace("True","true")
        json_acceptable_string = json_acceptable_string.replace(" d"," \"d")
        json_acceptable_string = json_acceptable_string.replace(")",")\"")
        json_acceptable_string = json_acceptable_string.replace("L,",",")
        json_acceptable_string = json_acceptable_string.replace("L}","}")

        # print(json_acceptable_string)
        d = json.loads(json_acceptable_string)
        if 'startTime' in d.keys():
            d['startTime'] = strToDate(d,'startTime')
        if 'endTime' in d.keys():
            d['endTime'] = strToDate(d,'endTime')
        if 'startGameTime' in d.keys():
            d['startGameTime'] = strToDate(d,'startGameTime')
            #exit(0)
        # print(s.id)
        complete_data[s.id] = d
        id[s.id] = curr.split("workerId': u'")[1]
        id[s.id] = id[s.id].split("'")[0]
        secreteKey[s.id] = curr.split("secretKey': u'")[1]
        secreteKey[s.id] = secreteKey[s.id].split("'")[0]

        
    # print workIds
    # exit(0)
    # print(complete_data['e5595675-5a0f-474c-8054-17d1fbde990a'])
    output = []
    # keep track of how many docs are there
    count = 0
    bad_ids = ["b151cf91-7805-4ee7-93a2-78a5cb706aad-495","2d8e511f-8963-41fb-8b43-1151ff94aebb-tutorial-21","2d8e511f-8963-41fb-8b43-1151ff94aebb-tutorial-28","c74151a6-1b88-4e92-8f0f-a48f11866e37-462"]
    # this is fore excel
    workers = []
    ids = []
    currentDics = []
    currentIDs = []
    displayStates = []
    id_count = 1
    worker_IDs = []
    prev_ID = ""
    doc_IDs = []
    UUIDs = []
    for doc in docs:
        currentID = str(doc.id)
        cut = -1
        if "-" in currentID:
            while currentID[cut] is not "-":
                cut -= 1
            #    get rid of 5da59b46-3634-4d39-9091-59a80cb0a0dd-10 tail "-10"
            currentID = currentID[0:cut]
            if prev_ID!=currentID and prev_ID!="":
                id_count += 1
            if currentID in workIds:
                # now get displayStateIndex
                # print str(doc.to_dict())
                # let's get action and rationalization
                currentDic = str(doc.to_dict()).split(": ")[3]
                # get rid of L and }
                #print(str(doc.to_dict()).split(": ")[3])
                #exit(0)
                currentDic = currentDic.split("L")[0]
                # TODO: use this newID to get the corresponding state from displayState collection
                newID = currentID + "-" + currentDic
                if newID not in bad_ids and 'tutorial' not in newID:
                    count += 1
                    workers.append(id[currentID])
                    outputSecreteKey.append(secreteKey[currentID])
                    actions.append(str(doc.to_dict()).split("u'action': ")[1].split("L")[0])
                    rationalizations.append(str(doc.to_dict()).split("u'rationalization': u")[1].split(",")[0])
                    ids.append(newID)
                    currentDics.append(currentDic)
                    currentIDs.append(currentID)
                    displayStates.append(int(currentDic))
                    worker_IDs.append(id_count)
                    doc_IDs.append(newID)
                    if 'startTime' in complete_data[currentID].keys():
                        start_times.append(complete_data[currentID]['startTime'].strftime('%x %X'))
                    else:
                        start_times.append("Filler Time")
                    if 'endTime' in complete_data[currentID].keys():
                        end_times.append(complete_data[currentID]['endTime'].strftime('%x %X'))
                    else:
                        end_times.append("Filler Time")
                    if 'startGameTime' in complete_data[currentID].keys():
                        game_start_times.append(complete_data[currentID]['startGameTime'].strftime('%x %X'))
                    else:
                        game_start_times.append("Filler Time")
                    finished_game.append(complete_data[currentID]['wonGame'])
                    finished_study.append(complete_data[currentID]['finished'])
                    # UUIDs.append(currentID)
            prev_ID = currentID

    for i,newID in enumerate(ids):
        displayCollection = db.collection(u'displayStates-turk').document(newID)
        states = displayCollection.get()
        curr = states.to_dict()
        # print curr
        output.append(curr)

        # also we want next second display state
        nextNum = int(currentDics[i]) + 1
        nextStateID = currentIDs[i]+ "-" + str(nextNum)
        nextDisplayCollection = db.collection(u'displayStates-turk').document(nextStateID)
        next = nextDisplayCollection.get().to_dict()
        nextState.append(next)

    analyseRationalizations(rationalizations)
            # print(u'{} => {}'.format(states.id, states.to_dict()))
    #print(count)    
    with open('log_file.json', 'w') as outfile:
        out = {"displayStates":output}
        # json.dump("displayStates:", outfile)
        json.dump(out, outfile)
    with open('log_fileNext.json', 'w') as outfile:
        out = {"displayStates":nextState}
        # json.dump("displayStates:", outfile)
        json.dump(out, outfile)

    ###############This part is to generate a excel file
    # we dont want the last line of doc
    # print(displayStates)
    # exit(0)
    count -= 1
    # print count
    # change actions to left right up down
    for i in range(0, len(actions)):
        if actions[i] == "1":
            actions[i] = "Left"
        if actions[i] == "2":
            actions[i] = "Right"
        if actions[i] == "3":
            actions[i] = "Up"
        if actions[i] == "4":
            actions[i] = "Down"
        if actions[i] == "8":
            actions[i] = "Wait"
    # print actions
    clock = 0
    current_images = []
    next_images = []
    data = []
    images = []
    while clock < count: 
        c_im = Image.open("Screenshot_" + str(clock) + ".png")
        temp1 = c_im.copy()
        c_im.close()
        current_images.append(temp1)
        n_im = Image.open("NextScreenshot_" + str(clock) + ".png")
        temp2 = n_im.copy()
        n_im.close()
        next_images.append(temp2)
        clock+=1
    clock = 0
    print(current_images)
    print(next_images)
    while clock < count: 
        d = dataset()
        d.doc_id = displayStates[clock]
        d.worker_ID = worker_IDs[clock]
        d.current_image = current_images[clock]
        d.next_image = next_images[clock]
        data.append(d)
        clock+=1
    clock = 0
    data = sorted(data, key = sort_key)
    while clock < count: 
        # fn1 = os.path.join(os.path.dirname(__file__), 'All_Images/Current_State/Screenshot_' + str(clock) + '.png')
        data[clock].current_image.save("D:/UnityFrogger_Screenshot/ScreenshotFrogger/cSharpTest/cSharpTest/bin/screenshotFireStore/All_Images/Current_State/Screenshot_" + str(clock) + ".png")
        # fn2 = os.path.join(os.path.dirname(__file__), 'All_Images/Next_State/NextScreenshot_' + str(clock) + '.png')
        data[clock].next_image.save("D:/UnityFrogger_Screenshot/ScreenshotFrogger/cSharpTest/cSharpTest/bin/screenshotFireStore/All_Images/Next_State/NextScreenshot_" + str(clock) + ".png")
        # print("inside")
        clock+=1
    workbook = xlsxwriter.Workbook('Turk_Master_File.xlsx')
    worksheet = workbook.add_worksheet()
    cell_format = workbook.add_format()
    cell_format.set_text_wrap()
    worksheet.set_column('B:B', 35)
    worksheet.set_column('A:A', 35)
    worksheet.set_column('D:D', 35)
    worksheet.set_column('F:F', 35, cell_format)
    worksheet.set_column('E:E', 20, cell_format)

    worksheet.set_default_row(150)
    worksheet.set_row(0, 18)

    clock = 0
    # print count
    worksheet.write('A1', 'Worker ID')
    worksheet.write('B1', 'Display State')
    worksheet.write('C1', 'Action')
    worksheet.write('D1', 'Next Display State')
    worksheet.write('E1', 'Rationalization', cell_format)
    worksheet.write('F1', 'Secret Key')
    worksheet.write('H1', 'Display State Id')
    worksheet.write('G1', 'Worker Number')
    worksheet.write('I1', 'Document ID')
    worksheet.write('J1', 'Won Game')
    worksheet.write('K1', 'Finished Study')
    worksheet.write('L1', 'Start Time')
    worksheet.write('M1', 'End Time')
    worksheet.write('N1', 'UUID')    
    

    clock = 0
    while clock < count:
        worksheet.write('A' + str(clock + 2), workers[clock])
        worksheet.insert_image('B' + str(clock + 2), "Screenshot_" + str(clock) + ".png", {'x_scale': 0.5, 'y_scale': 0.5})
        worksheet.write('C' + str(clock + 2), actions[clock])
        worksheet.write('E' + str(clock + 2), rationalizations[clock])
        worksheet.write('F' + str(clock + 2), outputSecreteKey[clock])
        worksheet.insert_image('D' + str(clock + 2), "NextScreenshot_" + str(clock) + ".png",
                               {'x_scale': 0.5, 'y_scale': 0.5})
        worksheet.write('H' + str(clock + 2), displayStates[clock])
        worksheet.write('G' + str(clock + 2), worker_IDs[clock])
        worksheet.write('I' + str(clock + 2), doc_IDs[clock])
        worksheet.write('J' + str(clock + 2), finished_game[clock])
        worksheet.write('K' + str(clock + 2), finished_study[clock])
        worksheet.write('L' + str(clock + 2), start_times[clock])
        worksheet.write('M' + str(clock + 2), end_times[clock])
        worksheet.write('N' + str(clock + 2), currentIDs[clock])
        clock += 1
    workbook.close()

