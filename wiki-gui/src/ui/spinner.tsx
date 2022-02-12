
export default function Spinner() {
    return (
        <div className="bg-green-500 relative flex justify-center items-center h-8 gap-3">
            <div className="rounded animate-spin ease duration-300 w-4 h-4 border-2 border-white">
            </div>
            <div className="text-white font-bold">
                Detecting...
            </div>
        </div>
    )
}