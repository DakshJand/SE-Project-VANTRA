# VANTRA Test Cases

## Test Case 1 — Open the VANTRA dashboard

| Field | Details |
|---|---|
| Test Case # | 1 |
| Test Case Name | Open the VANTRA dashboard |
| Page | 1 of 1 |
| System | VANTRA |
| Subsystem | Dashboard |
| Designed by | Daksh Jand, Saksham, Raunak Mahajan |
| Design Date | 11/09/2026 |
| Executed by |  |
| Execution Date |  |
| Short Description | Check that the VANTRA dashboard opens and shows the camera map. |

**Pre-conditions**

- VANTRA is running and the demo data has been loaded.

| Step | Action | Expected System Response | Pass / Fail | Comment |
|---:|---|---|---|---|
| 1 | Open the VANTRA dashboard in a browser. | The VANTRA page opens. |  |  |
| 2 | Look at the map. | Camera markers are visible on the map. |  |  |

**Post-conditions**

The dashboard opens and displays the camera map.

---

## Test Case 2 — Search for a vehicle

| Field | Details |
|---|---|
| Test Case # | 2 |
| Test Case Name | Search for a vehicle |
| Page | 1 of 1 |
| System | VANTRA |
| Subsystem | Search |
| Designed by | Daksh Jand, Saksham, Raunak Mahajan |
| Design Date | 15/09/2026 |
| Executed by |  |
| Execution Date |  |
| Short Description | Search for a sample vehicle and view its results. |

**Pre-conditions**

- The dashboard is open and demo data has been loaded.
- Use the sample plate `KA01VX2026`.

| Step | Action | Expected System Response | Pass / Fail | Comment |
|---:|---|---|---|---|
| 1 | Type `KA01VX2026` into the search box. | The sample plate is entered. |  |  |
| 2 | Click **Go** or press Enter. | VANTRA shows results for the vehicle. |  |  |
| 3 | Select the vehicle result. | The vehicle trip appears on the map. |  |  |

**Post-conditions**

A sample plate can be searched and its trip displayed.

---

## Test Case 3 — View a demo vehicle trip

| Field | Details |
|---|---|
| Test Case # | 3 |
| Test Case Name | View a demo vehicle trip |
| Page | 1 of 1 |
| System | VANTRA |
| Subsystem | Demo / Trajectory |
| Designed by | Daksh Jand, Saksham, Raunak Mahajan |
| Design Date | 19/09/2026 |
| Executed by |  |
| Execution Date |  |
| Short Description | Open a prepared demo and check that the vehicle route is shown. |

**Pre-conditions**

- The dashboard is open and demo data has been loaded.

| Step | Action | Expected System Response | Pass / Fail | Comment |
|---:|---|---|---|---|
| 1 | Open the **Demo** tab. | The list of demo scenarios appears. |  |  |
| 2 | Choose the **Clean baseline** scenario. | The selected vehicle trip opens. |  |  |
| 3 | Look at the map and trip details. | The route and camera stops are shown. |  |  |

**Post-conditions**

The clean demo trip is easy to open and view.
