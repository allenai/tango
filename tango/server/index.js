fetch("/api/runlist", {
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
})
  .then((res) => res.json())
  .then((out) => {
    const table = document.createElement("table");
    document.getElementById("content").appendChild(table);
    table.className='table';
    const trHead = document.createElement("tr");
    table.appendChild(trHead);
    trHead.className='table-header';
    const thName = document.createElement("th");
    trHead.appendChild(thName);
    thName.className='header__item';
    thName.innerHTML = "Run Name";
    const thStartDate = document.createElement("th");
    trHead.appendChild(thStartDate);
    thStartDate.className='header__item';
    thStartDate.innerHTML = "Start Date";
    out
      .map(v => { return { name:v.name, startDate:dayjs(v.start_date) } })
      .sort((a, b) => b.startDate-a.startDate)
      .forEach((val) => {
        const tr = document.createElement("tr");
        table.appendChild(tr);
        tr.className='table-row';
        const tdName = document.createElement("td");
        tr.appendChild(tdName);
        tdName.className='table-data';
        const ahref = document.createElement("a");
        tdName.appendChild(ahref);
        ahref.setAttribute("href", `/run/${val.name}`);
        ahref.innerHTML = val.name;
        const tdStartDate = document.createElement("td");
        tr.appendChild(tdStartDate);
        tdStartDate.className='table-data';
        tdStartDate.innerHTML = val.startDate.format("YYYY-MM-DD h:mm:ss A");
    });
  })
  .catch((err) => {
    throw err;
  });
