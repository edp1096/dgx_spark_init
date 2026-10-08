// Registered only while the user grants access to shopping.naver.com.
(() => {
  if (window.__sparkTalkResumeDraft) return;
  const nativeConfirm = window.confirm;
  let active = true;
  const isResume = (message) =>
    /리뷰|작성\s*중|임시\s*저장/.test(message) &&
    /이어서\s*(?:작성|쓰)|계속\s*작성|작성.*계속/.test(message);
  const resumeConfirm = function (message) {
    if (active && isResume(String(message || ""))) {
      document.documentElement?.setAttribute(
        "data-sparktalk-resumed-draft",
        "true",
      );
      return true;
    }
    return nativeConfirm.call(window, message);
  };
  window.confirm = resumeConfirm;
  window.__sparkTalkResumeDraft = true;
  document.addEventListener(
    "sparktalk:revoke-resume",
    () => {
      active = false;
      if (window.confirm === resumeConfirm) window.confirm = nativeConfirm;
      delete window.__sparkTalkResumeDraft;
      document.documentElement?.removeAttribute("data-sparktalk-resumed-draft");
    },
    { once: true },
  );
})();
