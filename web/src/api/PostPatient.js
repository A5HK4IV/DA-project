export const postPatient = async (data) => {
    const RawResponse = await fetch("/api/post/new/patient", {
        method: 'POST', headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })

    const res = await RawResponse.json()

    return res
}