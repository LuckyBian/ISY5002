主要流成大該就是
圖片-->找到人以及關鍵點(keypoints.py)-->丟進去pose.py生成骨骼圖，存儲的骨骼圖直接在pose.py裡面的draw就存儲了-->然後在丟進去action_detect文件中的detect.py判斷是否摔倒-->框出來，寫標籤-->結合圖片

然後如果要輸出骨骼圖，是在openpose_model文件中的pose.py，其中的draw方法就是存骨骼圖的地方

基本上就是運行runOpenpose.py文件，然後更改輸入的東西的地方是在這個py文件最下面的函數detect_main中的arrgument中更改路徑，圖片可以是單張或是文件夾，視頻就要是指定視頻路徑，不能讀取多個視頻。更改好參數後在往下看可以看到那些關於讀數據的判斷。然後輸入的路徑指定好後，他就會存到frame_provider中，使用run_demo來進行骨骼圖。

如果要更改骨骼圖的存儲路徑，就是在pose.py文件，draw方法的最下面那邊。