const spamWords = ["free", "winner", "prize", "click", "urgent"];

document.getElementById("checkBtn").addEventListener("click", () => {
  const message = document.getElementById("message").value.toLowerCase();
  const found = spamWords.some(word => message.includes(word));

  document.getElementById("result").textContent =
    found ? "Possible spam detected." : "No demo spam keywords detected.";
});
