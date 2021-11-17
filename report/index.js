fetch("/api/runlist", {
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
})
  .then((res) => res.json())
  .then((out) => {
    out.forEach((val) => {
      const div = document.createElement("div");
      const ahref = document.createElement("a");
      ahref.setAttribute("href", `/run/${val}`);
      ahref.innerHTML = val;
      div.appendChild(ahref);
      document.getElementById("content").appendChild(div);
    });
  })
  .catch((err) => {
    throw err;
  });
