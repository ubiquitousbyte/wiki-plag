
export default function Spinner() {
    return (
        <div className="bg-transparent relative flex justify-center items-center h-8 gap-3">
            <div className="rounded animate-spin ease duration-300 w-4 h-4 border-2 border-green-500">
            </div>
            <div className="text-green-500 font-bold">
                Detecting...
            </div>
        </div>
    )
}