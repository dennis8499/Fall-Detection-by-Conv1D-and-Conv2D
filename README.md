# Fall Detection by Conv1D and Conv2D
 利用加速度計來蒐集人體動作的數值後，經過處理及轉換並交由Conv1D和Conv2D來判斷有無跌倒事件發生
1. 資料夾:實驗原始數據
	1. 新的數據_頻率10Hz:為學姊所收集的數據
	2. readFile.py:從學姊所收集的數據中，擷取所需要的資料種類並寫入(新的數據_頻率10Hz/new)資料夾中
	3. makeImage.py:將(新的數據_頻率10Hz/new)資料夾中的數據做成圖檔(預設為曲線圖)，並存在(新的數據_頻率10Hz/Image)資料夾中
	4. changeFileName.py:將(新的數據_頻率10Hz/Image)資料夾中各檔案的檔名更改為數字(神經網路訓練用)
2. 資料夾:實驗
	1. MyProgram.py(論文用):1D訓練、1D測試、2D測試、整個流程測試
	2. time_classification.h5:1D神經網路的權重
	3. ANN.py:比較論文1
	4. ANN.h5:比較論文的神經網路權重
	5. ANN2.py:比較論文2
	6. ANN2.h5:比較論文2的神經網路權重
	7. 2018-11-16~2019-06-01為自己收集的數據(論文中不能用)
	8. DataSet:訓練的數據，只分為兩類(原始:把接掛電話去掉、測試用:原始所有數據)
	9. result:測試後的結果都會出現在這
	10. 神經網路訓練結果、測試結果(新)、測試結果(舊)、圖檔:皆為測試結果(有不同時期的結果，不用參考)，若裡面有10個資料夾，則為5x2 fold-cross validation
3. 資料夾:圖像實驗
	1. MobileNetV2.py(論文用):利用MobileNetV2來進行2D訓練(學校電腦ram只有8G不夠，讀檔完就滿了)
	2. DataSet:各種圖像種類(皆分為兩類而已)
	3. 結果:各式測試結果，若裡面有10個資料夾，則為5x2 fold-cross validation
	
############################################################################################################
使用方法
1. 如何開始訓練程式
	1. 開啟命令提示字元cmd
	2. 輸入activate tensorflow(開啟anaconda所創造的環境)
	3. 到達你所要執行的程式所在的資料夾(實驗 or 圖像實驗)
	4. 執行程式(python MyProgram.py)
	
2. MyProgram.py
	1. 調整各項參數 
		Train_path = 'DateSet/兩變數/train_2 (原始)' #訓練資料來源
		Test_path = '2019-05-19-1.txt'				 #當初自己的測試資料來源(可以不用管)
		Image_Test_path = 'result/'					 #結果檔案出來的位置
		windows = 50                                 #當初自己的測試資料每筆資料相隔的大小(可以不用管)
		NUM_CLASS = 2								 #分為兩類
		height = 173 #345 173(50%) 87(25%) 173       #圖像大小
		weight = 270 #460 270(50%) 115(25%) 230      #圖像大小
		Alpha = 0.9								     #低通濾波常數
		CONV1D_BATCH_SIZE = 32						 #1D神經網路訓練時的BATCH_SIZE
		CONV1D_EPOCHS = 1000                         #1D神經網路訓練的次數
	2. 訓練神經網路所需的Function
		trainX, trainY, filenames = load_data()	#讀取數據
		trainX_FFT = Transform_FFT(trainX)	    #數據轉換
		trainX_FFT_50, trainY_FFT_50, trainX_FFT_100, trainY_FFT_100 = splid_data_50(trainX_FFT, trainY) #5x2 fold cross-validation
		origin_trainX, origin_trainY, trainX, trainY, validX, validY, testX, testY = splid_data(trainX_FFT_100, trainY_FFT_100) #再將訓練數據集分成7:2:1用於訓練
		Neural_Network(trainX, trainY, validX, validY, testX, testY) #訓練
	3. 驗證所訓練的神經網路權重所需的Function
		trainX, trainY, filenames = load_data()	#讀取數據
		trainX_FFT = Transform_FFT(trainX)	    #數據轉換
		trainX_FFT_50, trainY_FFT_50, trainX_FFT_100, trainY_FFT_100 = splid_data_50(trainX_FFT, trainY) #5x2 fold cross-validation
		trainX_50, trainY_50, trainX_100, trainY_100 = splid_data_50(trainX, trainY) #原始數據切成50:50分別用於訓練及測試數據集 
		filenames_50, filenames_100 = splid_data_Filenames(filenames) #檔名切成50:50分別用於訓練及測試數據集
		model_predict_50(trainX_FFT_50, trainX_50, trainY_FFT_50, filenames_50) 
		#驗證整個流程，若先不測試2D數據則到model_predict_50()並將makePlot()以後的內容進行註解
	4. 注意事項
		1. 神經網路訓練與測試的資料要注意是否不同，訓練用A數據、測試要用B數據
		2. 整個流程訓練與測試要注意makePlot()所製作的圖像是與在MobileNetV2.py裡面所訓練的圖像是一樣的
		3. 注意各項參數的調整與變動
3. MobileNetV2.py
	1. 調整各項參數
		BATCH_SIZE = 64 #每次跑多少大小
		EPOCHS = 200 #跑幾次
		CLASS_NUM = 2 #分幾類
		height = 173 #345 173(50%) 87(25%) 173
		weight = 270 #460 270(50%) 115(25%) 230
		depth = 3
		
		Train_path = 'DateSet/train_2(XYZ_3_X255_None_lower)'  #訓練資料來源
	2.  訓練神經網路所需的Function
		trainX, trainY = load_data(Train_path)
		trainX_50, trainY_50, trainX_100, trainY_100 = splid_data_50(trainX, trainY) #5x2 fold cross-validation
		trainX, trainY, validX, validY, testX, testY  = split_data(trainX_50, trainY_50) #再將訓練數據集分成7:2:1用於訓練
		Neural_Network_MobileNetV2(trainX, trainY, validX, validY, testX, testY)
	4. 注意事項
		1. 神經網路訓練與測試的資料要注意是否不同，訓練用A數據、測試要用B數據
		2. 圖像訓練要記好是用哪一種圖像來訓練的
		3. 注意各項參數的調整
		4. 建議ram要到16g