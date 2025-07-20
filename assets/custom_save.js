// assets/custom_save.js

function attachSaveHtmlBtn() {
    var btn = document.getElementById('save-html-btn');
    if (btn && !btn.dataset.listenerAttached) {
        btn.onclick = function() {
            alert('버튼 클릭됨!');
            var html = document.documentElement.outerHTML;
            var blob = new Blob([html], {type: 'text/html'});
            var a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'snapshot.html';
            a.click();
        };
        btn.dataset.listenerAttached = "true";
        console.log('save-html-btn 이벤트 연결됨');
    }
}

// 최초 시도
attachSaveHtmlBtn();

// 이후 DOM 변화 감지
var observer = new MutationObserver(function(mutations) {
    attachSaveHtmlBtn();
});
observer.observe(document.body, { childList: true, subtree: true });

console.log('custom_save.js loaded'); 
