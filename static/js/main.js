const fileInput = document.querySelector('input[type="file"]');

if (fileInput) {
    fileInput.addEventListener("change", function () {
        const file = this.files[0];

        if (file) {
            const reader = new FileReader();

            reader.onload = function (e) {
                let img = document.createElement("img");
                img.src = e.target.result;
                img.style.width = "150px";
                img.style.marginTop = "10px";

                const old = document.querySelector(".preview-upload");
                if (old) old.remove();

                img.classList.add("preview-upload");
                fileInput.parentElement.appendChild(img);
            };

            reader.readAsDataURL(file);
        }
    });
}