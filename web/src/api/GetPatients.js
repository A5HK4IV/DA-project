export const getPatients = async () => {
    const RawResponse = await fetch("/api/get/patients")

    const res = await RawResponse.json()

    return res
}