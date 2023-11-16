document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('detectButton').addEventListener('click', function () {
        document.getElementById('detectButton').style.display = 'none';
        document.getElementById('status').style.display = 'block';

        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            let currentTab = tabs[0];
            if (!currentTab) {
                console.error('No active tab found');
                return;
            }

            let youtubeUrl = currentTab.url;

            fetch('http://cosmetic-detector.shop:8000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ youtube_link: youtubeUrl })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.message === "No cosmetic product found.") {
                        document.getElementById('status').textContent = "No cosmetic product found.";
                    } else {
                        updateUI(data);
                    }
                    document.getElementById('status').style.display = 'none';
                    document.getElementById('info').style.display = 'block';
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('status').textContent = 'Error fetching data';
                });
        });
    });
});

function updateUI(data) {
    let infoContainer = document.getElementById('info');
    infoContainer.innerHTML = ''; // 이전 결과 초기화

    data.forEach(product => {
        let productDiv = document.createElement('div');
        productDiv.innerHTML = `
            <div>
                <img src="${product.product_photo_url}" alt="Product Image" />
                <div>
                    <h3>${product.brand_name}</h3>
                    <p>${product.product_name}</p>
                    <p>Price: ${product.product_price}</p>
                </div>
            </div>
        `;

        productDiv.addEventListener('click', () => {
            window.open(product.product_detail_url, '_blank');
        });

        infoContainer.appendChild(productDiv);
    });
}
