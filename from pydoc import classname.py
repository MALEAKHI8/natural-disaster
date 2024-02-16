def get_class(model_path, labels_path, image_path):
    np.set_printoptions(surpass=True)
    model = load_model(model_path, compile=False)
    class_names = open (labels_path , "r", encodings="utf").readlines()
    data = np.ndarray(shape(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")

    size = (224,224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(Image)
    
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data(0) == normalized_image_array

    prediction = model_predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confident_score = prediction[0][index]
    text ='prediksi' (class_name[2:1])(confident_score)
    if class_name[2:-1] == "badai petir":
        text = text + '\n rekomendasi kamu tetap di dalam rumah'


    return text