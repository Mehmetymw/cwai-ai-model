document.addEventListener("DOMContentLoaded", function () {
    const tabs = document.querySelectorAll(".tab-btn"); // Tüm tab butonlarını seç
    const contents = document.querySelectorAll(".tab-content"); // Tüm tab içeriklerini seç

    // Sekme butonlarına tıklama olayını ekle
    tabs.forEach(tab => {
        tab.addEventListener("click", function () {
            const selectedTab = this.getAttribute("data-tab"); // Tıklanan sekmenin id'sini al

            // Aktif tab'ı ve içeriği göster
            tabs.forEach(button => button.classList.remove("active")); // Tüm tab butonlarından active sınıfını kaldır
            this.classList.add("active"); // Tıklanan butona active sınıfını ekle

            // Sekme içeriklerini kontrol et ve göster
            contents.forEach(content => {
                if (content.id === selectedTab) {
                    content.classList.remove("hidden"); // İlgili içeriği göster
                } else {
                    content.classList.add("hidden"); // Diğer içerikleri gizle
                }
            });
        });
    });

    // Sayfa yüklendiğinde varsayılan olarak ilk sekmeyi aktif et
    tabs[0].click();

    // Görüntü önizlemesi yapmak için
    function previewImage(file, section) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement("img");
            img.src = e.target.result;
            img.style.maxWidth = "100%";
            img.style.maxHeight = "100%";
            
            const previewId = section + '-preview';
            const previewElement = document.getElementById(previewId);
            previewElement.innerHTML = ''; // Önceki görüntüleri temizle
            previewElement.appendChild(img);
            previewElement.classList.remove("hidden");

            // Gönder butonunu göster
            document.getElementById(section + '-submit').classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    }

    // Dosya yükleme event'leri
    document.getElementById("ocr-file").addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) previewImage(file, 'ocr');
    });

    document.getElementById("food-file").addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) previewImage(file, 'food');
    });

    document.getElementById("waste-file-before").addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) previewImage(file, 'waste');
    });

    document.getElementById("waste-file-after").addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) previewImage(file, 'waste');
    });

    // API'ye veri gönderme
    function handleSubmit(section, endpoint) {
        const file = document.getElementById(section + '-file').files[0];
        if (!file) return;

        showLoading(section);

        const formData = new FormData();
        formData.append("file", file);

        fetch(endpoint, {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(result => showResults(section, result))
        .catch(error => {
            console.error("Error submitting data:", error);
            hideLoading(section);
        });
    }

    document.getElementById("ocr-submit").addEventListener("click", () => {
        handleSubmit("ocr", '/ocr-endpoint');
    });

    document.getElementById("food-submit").addEventListener("click", () => {
        handleSubmit("food", '/food-detection-endpoint');
    });

    document.getElementById("waste-submit").addEventListener("click", () => {
        handleSubmit("waste", '/waste-tracking-endpoint');
    });

    // Yükleniyor animasyonu
    function showLoading(section) {
        document.getElementById(section + '-loading').classList.remove("hidden");
        document.getElementById(section + '-submit').classList.add("hidden");
    }

    function hideLoading(section) {
        document.getElementById(section + '-loading').classList.add("hidden");
        document.getElementById(section + '-submit').classList.remove("hidden");
    }

    // Sonuçları göster
    function showResults(section, result) {
        hideLoading(section);
        const resultElement = document.getElementById(section + "-results");
        resultElement.classList.remove("hidden");

        if (section === "ocr") {
            resultElement.innerHTML = `<p>${result.text}</p>`;
        } else if (section === "food") {
            resultElement.innerHTML = `<p>Detected Food: ${result.food}</p>`;
        } else if (section === "waste") {
            document.querySelector('#waste-results .progress').style.width = result.wastePercentage + "%";
            resultElement.innerHTML += `
                <p>Atık tahmini: ${result.waste}</p>
                <p>CO2 Emisyonu: ${result.co2_emission}</p>
            `;
        }
    }
});
