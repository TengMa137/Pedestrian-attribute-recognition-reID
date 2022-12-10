from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from resnet import res_net, resnet
from datasets import ImagesData, testData, testimage
import makeCSV
from evaluate import get_AP, get_CMC
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans

makeCSV.create_csv("classification_test.csv")

transform = transforms.Compose([
        transforms.RandomHorizontalFlip()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
train_set = ImagesData("annotations_train.csv", "./dataset/train")#, transform=transform)#datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
val_set = ImagesData("annotations_train.csv", "./dataset/val")#datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val']) 
test_set = testData("./dataset/test")  
img_path = "./dataset/test/000014.jpg"
#test_img = testimage(img_path)
train_allset = ImagesData("annotations_train.csv", "./train")

#valnum = 151, trainnum = 600
train_loader = DataLoader(train_set, batch_size = 128, shuffle = True, num_workers = 8)
val_loader = DataLoader(val_set, batch_size = 64, shuffle = False, num_workers = 4)
test_loader = DataLoader(test_set, batch_size = 64, shuffle = False, num_workers = 4)
#img_loader = DataLoader(test_img, batch_size=1)
train_alloader = DataLoader(train_allset, batch_size = 128, shuffle = True, num_workers = 8)

# training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#load model
model = resnet()#resnet()
model.load_state_dict(torch.load('resnet50_all'))#,strict=False)
#model = res_net()
#model.load_state_dict(torch.load('resnet50_att'),strict=False)
model.to(device)#eval() #= model.load_state_dict(model_data)


EPS = 1e-6
#def
weight = val_set.weight.to(device) #print(train_set.weight) 
criterion = torch.nn.BCEWithLogitsLoss(pos_weight = weight)

lr_adjust = {10: 0.01, 50: 0.001, 100: 0.0001}
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001,momentum=0.8, nesterov=True, weight_decay=0.0001) #
#optimizer = torch.optim.Adam(model.parameters())
#optimizer = torch.optim.AdamW(model.parameters())
train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []
#train_epochs_loss1 = []

def train(epochs, data_loader):    
    #train_epoch_loss = []
    for epoch in range(epochs):
        model.train()
        for idx, (input, labels) in enumerate(data_loader,0):
            input, labels = input.to(device), labels.to(device)
            #print(input.size())
            labels = labels.squeeze().float()
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())  
            if idx%len(data_loader)==0:
                print("epoch={}, loss={:.3f}".format(epoch, loss.item()))      
        train_epochs_loss.append(np.average(train_loss))   
        '''
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))
        '''

    #torch.save(model.state_dict(),'resnet50_att')

def validate():    
    with torch.no_grad():
        model.eval()
        classnum = 32
        
        tp = np.zeros(classnum)
        tn = np.zeros(classnum)
        labs = np.zeros(classnum)
        pred = np.zeros(classnum)
        precision = np.zeros(classnum)
        recall = np.zeros(classnum)       
        f1 = np.zeros(classnum)
        #N = len(val_set)
        #mA = np.zeros(classnum)

        for idx, (input, labels) in enumerate(val_loader):

            input, labels = input.to(device), labels.to(device)
            outputs = model(input)
            #labels = labels.squeeze().float()

            labels = labels.cpu().numpy().squeeze()
            outputs = outputs.cpu().numpy()
            outputs = np.where(outputs > 0, 1, 0)   
            #print(outputs.size())   
            
            for i in range(len(outputs)):
                if np.sum(outputs[i, 9:17])==0: outputs[i,26] = 1
                if np.sum(outputs[i, 17:26])==0: outputs[i,27] = 1
                #if (outputs[i, 1] == 1 or outputs[i, 2] == 1 or outputs[i, 0] == 1): outputs[i, 0] =  1
                for j in range(len(outputs[i])):
                    
                    if outputs[i][j] == 1 and labels[i][j] == 1: tp[j] += 1
                        
                    elif outputs[i][j] == 0 and labels[i][j] == 0: tn[j] += 1
            
            labs += labels.sum(0)#.astype(int)
            pred += outputs.sum(0)#.astype(int)
            #print("batch: ", idx)
            #print("labs: ", labs)
            #print("pred: ", pred)
            #print("tp: ", tp)

        precision = tp/pred #(tp+EPS)/(pred+EPS) 
        recall = tp/labs #(tp+EPS)/(labs+EPS)
        f1 = 2 * precision * recall / (precision + recall)
        #mA = 0.5 * (tp/pred + tn/(N - pred)) 
        
        np.set_printoptions(suppress=True, precision = 4)
        print('precision: ', precision, sum(precision[~np.isnan(precision)])/classnum)#len(precision[~np.isnan(precision)]))
        print('recall: ', recall, sum(recall[~np.isnan(recall)])/classnum)#len(recall[~np.isnan(recall)]))
        print("F1 score: ", f1, sum(f1[~np.isnan(f1)])/classnum)#len(f1[~np.isnan(f1)]))
        #print("mA: ", mA, mA[~np.isnan(mA)].sum()/len(mA[~np.isnan(mA)])) 
    return f1

def test():
    with torch.no_grad():
        model.eval()
        out = np.zeros([1,32])#out = torch.empty(1,32)
        #names = []
        #result = []
        for idx, (img, img_name) in enumerate(test_loader,0):#for img, img_name in test_loader:
            img = img.to(device)
            output = model(img).cpu().numpy()
            out = np.append(out, output, axis=0)#out = torch.cat((out,output),dim=0)
            #names = torch.cat((names, img_name),dim=0)
            #print(out.size(),names.size())
            #print(output.shape)

            #img_name = img_name#.cpu().numpy()
            #print(len(img_name))
            #names = np.append(names, img_name, axis=0)#output = output.cpu().numpy() 
            #print(type(img_name))
            output[:, 9:17] = (output[:, 9:17] == np.max(output[:, 9:17], axis=1, keepdims=1)).astype(int) #non maximum depress
            output[:, 17:26] = (output[:, 17:26] == np.max(output[:, 17:26], axis=1, keepdims=1)).astype(int)
            output[:, 28:32] = (output[:, 28:32] == np.max(output[:, 28:32], axis=1, keepdims=1)).astype(int)
            
            output = np.where(output > 0 , 1, 0)
            if np.sum(output[:, 9:17])==0: output[26] = 1
            if np.sum(output[:, 17:26])==0: output[27] = 1
            for i in range(len(output)):
                
                #if (output[i, 1] == 1 or output[i, 2] == 1 or output[i, 0] == 1): output[i, 0] =  1
                outimg = np.append(img_name[i],output[i])
                
                makeCSV.append_list_as_row('classification_test.csv',outimg)#[img_name[i], torch.split(output[i], torch.cat([a,b],0), dim=0)])# = [img_name, output]
        
        out = out[1:,]
        #result = np.ones([len(out),1])
        #for i in range(len(out)):
            #if np.sum(np.abs(out[i, [3,4,5,29]]))>10: result[i]=1 
            #if np.max(out[i,11:30])<1: result[i]=0

        #print(result.shape)
        
        #out = out[:, [3,4,5,29]]
        #out = np.sum(np.abs(out[:, [3,4,5,29]]),axis=1,keepdims=True)#.squeeze()
    
        #kmeans = KMeans(n_clusters=2, random_state=0).fit(out) 
        #result = kmeans.labels_
        #print(sum(result))
        #print(sum(result),result.shape) #cluster_test

        makeCSV.get_csv('classification_test.csv')
        df = pd.read_csv('classification_test.csv')
        #df1 = pd.DataFrame({"filename":names}).astype(str)
        #df1["kind"] = result.astype(int)
        #df = df.iloc[] 
        #df = pd.merge(df1,df, on='filename')
        df.to_csv('classification_test.csv', index=False)
        print(df)    

def reid(gallerypath, querypath, test = 0):
    
    train_reid = testData(gallerypath) #"./dataset/train_reid" when not testing
    val_reid = testData(querypath)#"./dataset/val_reid"
    train_reid_loader = DataLoader(train_reid, batch_size = 128, shuffle = False, num_workers = 8)
    val_reid_loader = DataLoader(val_reid, batch_size = 128, shuffle = False, num_workers = 8)
    trainout = np.zeros([1,32])
    valout = np.zeros([1,32])
    namest = np.zeros([1])
    namesv = np.zeros([1])
    with torch.no_grad():
        model.eval()    
        for idx, (img, names) in enumerate(train_reid_loader):
                img = img.to(device)
                outputs = model(img)
                outputs = outputs.cpu().numpy()  
                trainout = np.append(trainout, outputs, axis=0)
                if test == 0:
                    names = [i.split('_')[0] for i in names] 
                namest = np.append(namest, names, axis=0) 
        for idx, (img, names) in enumerate(val_reid_loader):
                img = img.to(device)
                outputs = model(img)
                outputs = outputs.cpu().numpy()  
                valout = np.append(valout, outputs, axis=0) 
                if test == 0:
                    names = [i.split('_')[0] for i in names] 
                namesv = np.append(namesv, names, axis=0) 
    trainout = trainout[1:,]
    valout = valout[1:,]
    namesv = namesv[1:,]
    namest = namest[1:,]
        
    #f1 = validate()
    w = np.ones(32)
    #w[~np.isnan(f1)] = f1[~np.isnan(f1)]#np.exp(f1[~np.isnan(f1)])
    #w[np.isnan(f1)] = 1
    
    result = []#np.zeros(len(namesv),dtype=object)
    
    for i in range(len(namesv)):
        pred = []
        for j in range(len(namest)):
            #v = np.where(valout[i] > 0 , 1, 0)
            #t = np.where(trainout[j] > 0 , 1, 0)
            pred.append(sum((valout[i] - trainout[j])**2 * w)/32)#  np.max(valout[i] - trainout[j])**2)
            #if (v[np.where(f1>0.9)] == t[np.where(f1>0.9)]).all(): 
            #    pred.append(sum((valout[i] - trainout[j])**2 * w)) #/weight np.linalg.norm(valout[i], trainout[j])
            #else: pred.append(1e10)
        #pred = np.array(pred)
        #print(namest[np.argsort(pred)])
        #result[i] = namest[np.argsort(pred)]
        result.append(namest[np.argsort(pred)])
        r = namest[np.argsort(pred)]
        pred = sorted(pred)#.sort() not working
        #print(pred[20])
        
        if test:
            for p in range(len(pred)):
                #if pred[p]-pred[0]>9:
                #if (pred[p+3]-pred[p+1])/2 > (pred[p+1]-pred[1])/p and pred[1]<500:
                if pred[p]>12: 
                    r = r[:p]
                    #if p == 1: r = r[0]
                    #else: r = r[:p]
                    break
            #if len(r) == 0: r = r[0]
            #if len(r)==len(namest[np.argsort(pred)]): r = r[0]
            f = open("reid_test.txt",'a')
            f.write(namesv[i])
            f.write(": ")
            for i in r: 
                
                if len(r) ==1: f.write(r[0])
                elif i == r[-1]: f.write(i)
                else: f.write(i+',')
            
            f.write('\n')
            #np.savetxt("reid_test.txt", r)#namesv[i])           
            #f.close()
    #print(len(result))
    if test == 0:
        top1acc = []
        top5acc = []
        AP = []
        for i in range(len(namesv)):
            top1, top5 = get_CMC(namesv[i], result[i])
            ap = get_AP(namesv[i], result[i])
            top1acc.append(top1)
            top5acc.append(top5)
            AP.append(ap)
        mAP = sum(AP)/len(namesv)
        top1_acc = sum(top1acc)/len(namesv)
        top5_acc = sum(top5acc)/len(namesv)
        print("top1 accuracy = {}, top5 accuracy = {}, mAP = {}".format(top1_acc, top5_acc, mAP))       


#def plotloss(loss1, loss2):#train_loss, 
    #plt.figure(figsize=(12,4))
    #plt.subplot(121)
    #plt.plot(train_epochs_loss[1:],'-o',label="train_loss")
    #plt.plot(train_loss[:])
    #plt.title("train_epochs_loss")
    #plt.subplot(122)
    #plt.plot(loss1[1:], '-o', label="SGD_loss")# loss2[1:], 'r--')
    #plt.plot(loss2[1:], '-o', label="AdamW_loss")#
    #plt.plot(valid_loss[1:],label="valid_loss")
    #plt.title("valid_loss")
    #plt.legend()
    #plt.show()
#plotloss(train_epochs_loss, train_epochs_loss1)
'''
def testAimg():
    with torch.no_grad():
        model.eval()
        for img in img_loader:
            
            img = img.to(device)
            output = model(img)#.cpu().numpy()
            #print(img)
            print(output)
            im = Image.open(img_path)
            plt.imshow(im)
            plt.show()          
'''
#train(20, train_loader)
#validate()
#test()
#reid(gallerypath="./dataset/train_reid", querypath="./dataset/val_reid")    #validation
reid(gallerypath="./dataset/test", querypath="./dataset/queries", test = 1) #test



             


